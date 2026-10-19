"""Incrementally index complete validation history from existing, user-owned logs."""
import json
import copy
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time

METRICS = {'psnr':'max','ssim':'max','lpips':'min','l1':'min','l2':'min',
           'mae':'min','mse':'min','rmse':'min','loss':'min','fid':'min'}
NUMBER = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'


def metric_name(name):
    name = name.lower().replace('/','_').replace('-','_').replace('.','_')
    name = re.sub(r'^(?:val(?:idation)?|eval(?:uation)?|test)_','',name)
    return name if name in METRICS or re.fullmatch(r'(?:psnr|ssim)_(?:rgb|y)',name) else None


def direction(name):
    return METRICS.get(name.split('_')[0],'max')


def numeric(value):
    if isinstance(value,bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError,OverflowError):
        return None


def parse_validation(line, context, event_id, dedicated=False):
    line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line).strip()
    begin = line.find('{')
    record = None
    if begin >= 0:
        try:
            record = json.loads(line[begin:])
        except ValueError:
            pass
    if isinstance(record,dict):
        for key in ('step','global_step','iteration','iter','epoch'):
            value=record.get(key)
            if isinstance(value,int) and not isinstance(value,bool):
                context['epoch' if key=='epoch' else 'step']=value
        if isinstance(record.get('phase'),str):
            context['phase']=record['phase']
        phase=str(record.get('phase',context.get('phase','')))
        dataset=str(record.get('dataset',record.get('val_dataset',record.get('dataset_name','未标注验证集'))))
        variant=record.get('nfe',record.get('inference_steps',record.get('sampling_steps')))
        scope=str(record.get('weight_source',''))
        events=[]

        def collect(values, force=False, nfe=variant, dataset_name=dataset, label=''):
            if not isinstance(values,dict):
                return
            metrics={}
            for key,value in values.items():
                name=metric_name(str(key))
                explicit=bool(re.match(r'^(?:val(?:idation)?|eval(?:uation)?|test)[_/.-]',str(key),re.I))
                number=numeric(value)
                if name and number is not None and (force or explicit):
                    metrics[name]=number
            if metrics:
                events.append(dict(dataset=dataset_name,phase=phase,nfe=nfe,scope=scope,label=label,
                                   step=context.get('step'),epoch=context.get('epoch'),metrics=metrics,
                                   timestamp=record.get('timestamp',record.get('time')),event_id=event_id))
            for key,value in values.items():
                if not isinstance(value,dict):
                    continue
                key_lower=str(key).lower()
                if key_lower=='by_nfe':
                    for nfe_key,nfe_values in value.items():
                        collect(nfe_values,True,str(nfe_key),dataset_name)
                elif key_lower in ('val','validation','eval','evaluation','test','metrics'):
                    collect(value,force or key_lower!='metrics',nfe,dataset_name)
                elif force:
                    collect(value,True,nfe,str(key))

        event_label=' '.join(str(record.get(key,'')) for key in ('event','mode','split','stage','phase')).lower()
        marked=dedicated or bool(re.search(r'val|eval|test',event_label))
        collect(record,marked)
        # A by_nfe table and its selected top-level aliases describe the same result.
        merged={}
        for event in events:
            key=json.dumps([event['dataset'],event['phase'],str(event['nfe']) if event['nfe'] is not None else '',event['scope']],ensure_ascii=False)
            if key not in merged:
                merged[key]=event
            else:
                merged[key]['metrics'].update(event['metrics'])
            merged[key]['key']=key
        return list(merged.values())

    # Common text logs, including BasicSR's multiline "Validation NAME" blocks.
    step=re.search(r'\b(?:step|iter(?:ation)?s?)\s*[:=\[]\s*([\d,]+)',line,re.I)
    epoch=re.search(r'\bepoch\s*[:=\[]\s*(\d+)',line,re.I)
    if step:
        context['step']=int(step.group(1).replace(',',''))
    if epoch:
        context['epoch']=int(epoch.group(1))
    header=re.search(r'\b(?:validation|validating|evaluation)\b\s*[:\-]?\s*([^:#\n]*)',line,re.I)
    if header:
        name=header.group(1).strip()
        context['text_dataset']=name if name and not re.search(r'psnr|ssim|step|epoch',name,re.I) else '未标注验证集'
        context['text_block']=event_id
    explicit=bool(re.search(r'\bval(?:idation)?[_ /-]|\beval(?:uation)?\b|\bvalidation\b',line,re.I))
    in_block='text_block' in context and bool(re.match(r'^\s*(?:#|psnr|ssim|lpips|l1|l2|mae|mse|loss)',line,re.I))
    if not (dedicated or explicit or in_block):
        if line and not header:
            context.pop('text_block',None)
        return []
    metrics={}
    for match in re.finditer(r'\b((?:(?:val|validation|test)[_/ ])?(?:psnr(?:_(?:rgb|y))?|ssim(?:_(?:rgb|y))?|lpips|l1|l2|mae|mse|rmse|loss))\s*[:=]\s*('+NUMBER+r')',line,re.I):
        name=metric_name(match.group(1).replace(' ','_'))
        value=numeric(match.group(2))
        if name and value is not None:
            metrics[name]=value
    if not metrics:
        return []
    nfe_match=re.search(r'\b(?:nfe|inference_steps)\s*[:=]\s*(\d+)',line,re.I)
    nfe=nfe_match.group(1) if nfe_match else None
    dataset=context.get('text_dataset','未标注验证集')
    phase=context.get('phase','')
    key=json.dumps([dataset,phase,str(nfe) if nfe is not None else '',''],ensure_ascii=False)
    return [dict(key=key,dataset=dataset,phase=phase,nfe=nfe,scope='',step=context.get('step'),epoch=context.get('epoch'),
                 metrics=metrics,event_id=context.get('text_block',event_id) if in_block else event_id)]


class ValidationIndex:
    def __init__(self, directory):
        self.path=Path(directory)/'validation-cache.json'
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.lock=threading.RLock()
        try:
            self.sources=json.loads(self.path.read_text())
        except (OSError,ValueError):
            self.sources={}

    @staticmethod
    def add(source, event, path, line_number, raw):
        group=source['groups'].setdefault(event['key'],dict(key=event['key'],dataset=event['dataset'],phase=event['phase'],
                    nfe=event['nfe'],scope=event['scope'],count=0,best={},history=[]))
        row={**event,'source':path,'line':line_number,'raw':raw[:8192]}
        # Text validation blocks can emit one metric per line. Merge the same event.
        if group.get('latest',{}).get('event_id')==event['event_id']:
            row={**group['latest'],**row,'metrics':{**group['latest']['metrics'],**row['metrics']},
                 'raw':(group['latest']['raw']+'\n'+row['raw'])[-65536:]}
            group['history'][-1]=row
        else:
            group['count']+=1
            group['history'].append(row)
            group['history']=group['history'][-200:]
        group['latest']=row
        for name,value in row['metrics'].items():
            previous=group['best'].get(name)
            mode=direction(name)
            if previous is None or (value>previous['value'] if mode=='max' else value<previous['value']) or previous['event_id']==row['event_id']:
                group['best'][name]={**row,'value':value,'direction':mode}

    def summary(self, path, budget=4*1024*1024, history=False, dedicated=None):
        if not path:
            return dict(groups=[],sources=[],scanning=False,errors=[],note='尚未关联可读取的验证日志')
        path=str(Path(path).resolve())
        source_key=json.dumps([path,dedicated])
        with self.lock:
            try:
                descriptor=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
                with os.fdopen(descriptor,'rb') as stream:
                    info=os.fstat(stream.fileno())
                    if info.st_uid!=os.getuid() or not stat.S_ISREG(info.st_mode):
                        raise ValueError('仅可索引当前用户拥有的普通日志文件')
                    fingerprint=f'{info.st_dev}:{info.st_ino}'
                    source=self.sources.get(source_key)
                    if not source or source['fingerprint']!=fingerprint or source['offset']>info.st_size:
                        source=dict(fingerprint=fingerprint,offset=0,lines=0,groups={},context={})
                        self.sources[source_key]=source
                    stream.seek(source['offset'])
                    data=stream.read(budget)
                    end=data.rfind(b'\n')+1
                    if end:
                        complete=data[:end]
                        position=source['offset']
                        is_dedicated=dedicated if dedicated is not None else bool(re.search(r'(?:^|[_./-])(?:val(?:idation)?|eval(?:uation)?|test)(?:[_./-]|$)',Path(path).stem,re.I))
                        for raw in complete.splitlines(keepends=True):
                            source['lines']+=1
                            text=raw.decode('utf-8',errors='replace')
                            for event in parse_validation(text,source['context'],position,is_dedicated):
                                self.add(source,event,path,source['lines'],text)
                            position+=len(raw)
                        source['offset']+=end
                        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.path.parent,delete=False) as temporary:
                            json.dump(self.sources,temporary,ensure_ascii=False)
                            name=temporary.name
                        os.replace(name,self.path)
                    scanning=source['offset']<info.st_size and len(data)>=budget
                groups=[{key:value for key,value in group.items() if history or key!='history'} for group in source['groups'].values()]
                return dict(groups=copy.deepcopy(groups),sources=[path],scanning=scanning,errors=[],
                            indexed_bytes=source['offset'],file_bytes=info.st_size,
                            note='历史最优按当前可用日志全历史计算；各指标独立取优，NFE、阶段和验证集分别统计')
            except (OSError,ValueError) as exc:
                return dict(groups=[],sources=[path],scanning=False,errors=[str(exc)])
