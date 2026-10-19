import json
import os
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation_metrics import ValidationIndex


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.log=self.root/'training.log'
        self.index=ValidationIndex(self.root/'cache')

    def tearDown(self):
        self.tmp.cleanup()

    def append(self,record):
        with self.log.open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    def test_full_history_best_survives_long_training_tail_and_restart(self):
        self.append({'step':100,'val_psnr':33,'val_ssim':.90,'val_l1':.04})
        self.append({'step':200,'val_psnr':32,'val_ssim':.92,'val_l1':.02})
        with self.log.open('a') as stream:
            stream.write(('training output padding '*100+'\n')*300)
        result=self.index.summary(self.log,history=True)
        group=result['groups'][0]
        self.assertEqual(group['best']['psnr']['value'],33)
        self.assertEqual(group['best']['psnr']['step'],100)
        self.assertEqual(group['best']['psnr']['metrics']['ssim'],.90)
        self.assertEqual(group['best']['ssim']['step'],200)
        self.assertEqual(group['best']['l1']['value'],.02)
        restarted=ValidationIndex(self.root/'cache')
        self.assertEqual(restarted.summary(self.log)['groups'][0]['best']['psnr']['step'],100)
        self.assertEqual(restarted.summary(self.log)['groups'][0]['count'],2)

    def test_separate_nfe_phase_and_dataset_and_deduplicate_aliases(self):
        self.append({'phase':'bridge','step':100,'inference_steps':1,'dataset':'SetA',
                     'val_psnr':30,'val_ssim':.9,'by_nfe':{'1':{'psnr':30,'ssim':.9},'5':{'psnr':32,'ssim':.92}}})
        self.append({'phase':'mean','step':100,'dataset':'SetA','val_psnr':34})
        self.append({'phase':'mean','step':100,'dataset':'SetB','val_psnr':35})
        result=self.index.summary(self.log)
        self.assertEqual(len(result['groups']),4)
        ones=next(g for g in result['groups'] if str(g['nfe'])=='1')
        self.assertEqual(ones['count'],1)
        self.assertEqual(ones['best']['psnr']['value'],30)

    def test_training_metrics_not_mistaken_for_validation(self):
        for name in ('training.log','latest.log'):
            path=self.root/name
            path.write_text('{"step":10,"loss":0.01,"psnr":40}\n')
            self.assertEqual(self.index.summary(path)['groups'],[])
        path=self.root/'validation.jsonl'
        path.write_text('{"step":10,"psnr":30,"ssim":0.9,"lpips":0.1}\n')
        group=self.index.summary(path)['groups'][0]
        self.assertEqual(group['best']['lpips']['direction'],'min')

    def test_text_multiline_block_and_text_validation(self):
        self.log.write_text('epoch: 1 iter: 1,000\nValidation SetA\n# psnr: 31.5 Best: 40.0 @ 300 iter\n# ssim: 0.91\n')
        group=self.index.summary(self.log,history=True)['groups'][0]
        self.assertEqual(group['count'],1)
        self.assertEqual(group['best']['psnr']['value'],31.5)
        self.assertEqual(group['best']['psnr']['metrics']['ssim'],.91)
        self.assertEqual(group['latest']['step'],1000)
        self.assertIn('# psnr',group['latest']['raw'])
        self.assertIn('# ssim',group['latest']['raw'])
        with self.log.open('a') as stream:
            stream.write('Validation step: 2000 PSNR: 32.1 SSIM: 0.92\n')
        result=self.index.summary(self.log)
        self.assertTrue(any(g['best']['psnr']['value']==32.1 for g in result['groups']))

    def test_incremental_partial_line_truncation_and_rotation(self):
        self.append({'step':100,'val_psnr':31})
        self.index.summary(self.log)
        with self.log.open('a') as stream:
            stream.write('{"step":200,"val_psnr":')
        self.assertEqual(self.index.summary(self.log)['groups'][0]['count'],1)
        with self.log.open('a') as stream:
            stream.write('32}\n')
        self.assertEqual(self.index.summary(self.log)['groups'][0]['count'],2)
        self.log.write_text('{"val_psnr":20}\n')
        self.assertEqual(self.index.summary(self.log)['groups'][0]['best']['psnr']['value'],20)
        replacement=self.root/'replacement'
        replacement.write_text('{"val_psnr":25}\n')
        os.replace(replacement,self.log)
        self.assertEqual(self.index.summary(self.log)['groups'][0]['best']['psnr']['value'],25)

    def test_scanning_budget_reports_incomplete_history(self):
        for step in range(100):
            self.append({'step':step,'val_psnr':step})
        first=self.index.summary(self.log,budget=250)
        self.assertTrue(first['scanning'])
        for _ in range(100):
            last=self.index.summary(self.log,budget=250)
            if not last['scanning']:
                break
        self.assertFalse(last['scanning'])
        self.assertEqual(last['groups'][0]['count'],100)
        self.assertEqual(last['groups'][0]['best']['psnr']['value'],99)

    def test_best_preserved_after_bounded_history_and_invalid_values_ignored(self):
        for step in range(250):
            self.append({'step':step,'val_psnr':50-step/100,'val_lpips':step/1000})
        self.append({'step':251,'val_psnr':float('nan'),'val_ssim':float('inf')})
        group=self.index.summary(self.log,history=True)['groups'][0]
        self.assertEqual(len(group['history']),200)
        self.assertEqual(group['best']['psnr']['step'],0)
        self.assertEqual(group['best']['lpips']['step'],0)
        self.assertNotIn('ssim',group['best'])

    def test_file_ownership_regular_file_checks_and_missing_source(self):
        self.assertEqual(self.index.summary(None)['groups'],[])
        self.assertTrue(self.index.summary(self.root/'missing')['errors'])
        fifo=self.root/'fifo'
        os.mkfifo(fifo)
        self.assertTrue(self.index.summary(fifo)['errors'])

    def test_explicit_validation_source_works_with_arbitrary_filename(self):
        self.log.write_text('{"step":100,"psnr":31,"ssim":0.91}\n')
        self.assertEqual(self.index.summary(self.log)['groups'],[])
        self.assertEqual(self.index.summary(self.log,dedicated=True)['groups'][0]['best']['psnr']['value'],31)


if __name__=='__main__':
    unittest.main()
