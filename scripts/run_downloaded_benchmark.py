"""CQ stages for a reproducible first-pass evaluation of every public downloaded group."""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from supervised.__main__ import parser as training_parser
from supervised.data import prepare_builtin,load
from supervised.engine import train,predict,file_hash
from supervised.transaction_prepare import prepare_transactions,prepare_elliptic

DATASETS=['sd-stock','sd-crypto','tstr','ulb','sapienza','elliptic','paysim','ibm_aml','tabformer']


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def run_dataset(dataset,args,out):
    started=time.monotonic()
    run=out/dataset
    if (run/'result.json').exists():
        return json.loads((run/'result.json').read_text())
    run.mkdir(parents=True,exist_ok=True)
    inputs=run/'inputs'
    source=Path('data/supervised_financial')
    if dataset in ('paysim','tabformer','ibm_aml'):
        prepare_transactions(dataset,source,inputs,args.train_cap,args.validation_cap,seed=args.seed)
    elif dataset=='elliptic':
        prepare_elliptic(source,inputs)
    else:
        prepare_builtin(dataset,source,inputs,history_length=32 if dataset in ('sapienza','ulb') else 0)
    prep=json.loads((inputs/'preparation.json').read_text()) if (inputs/'preparation.json').exists() else {}
    atomic_json(run/'preparation.json',prep)
    ntrain=len(load(inputs/'train.npz').X)
    batch=4 if dataset=='tstr' else min(32,ntrain)
    command=['train','--data-dir',str(inputs),'--output',str(run/'model'),
        '--backbone','NeoQuasar/Kronos-mini','--epochs',str(args.epochs),'--batch-size',str(batch),
        '--seed',str(args.seed),'--device',args.device,'--threads','2','--defer-test',
        '--threshold-policy','f1','--patience','3']
    if dataset=='tstr':command+=['--max-context','2048']
    trained=train(training_parser().parse_args(command))
    predict(argparse.Namespace(command='evaluate',checkpoint=str(run/'model/best.pt'),
        data=str(inputs/'test.npz'),output=str(run/'test_predictions.csv'),device=args.device,
        batch_size=batch,threads=2))
    evaluated=json.loads((run/'test_predictions.csv.metrics.json').read_text())
    table=pd.read_csv(run/'test_predictions.csv',usecols=['label','anomaly_probability'])
    known=table.label>=0
    precision,recall,thresholds=precision_recall_curve(table.loc[known,'label'],table.loc[known,'anomaly_probability'],drop_intermediate=True)
    bins=np.floor(recall*1000).astype(int)
    edges=np.flatnonzero(bins[1:]!=bins[:-1])
    operating=min(int(np.searchsorted(thresholds,evaluated['metrics']['threshold'])),len(precision)-1)
    selected=np.unique(np.r_[0,edges,edges+1,len(precision)-1,operating])
    atomic_json(run/'pr_curve.json',dict(precision=precision[selected].tolist(),recall=recall[selected].tolist(),
        prevalence=float(table.loc[known,'label'].mean())))
    rows=len(table)
    if prep.get('dataset')=='elliptic':unit='transaction node'
    elif dataset=='tstr':unit='whole sequence'
    elif dataset.startswith('sd-') or dataset=='sapienza':unit='time point'
    else:unit='transaction'
    result=dict(dataset=dataset,status='complete',unit=unit,seed=args.seed,
        epochs_executed=trained['epochs_executed'],best_epoch=trained['best_epoch'],
        validation=trained['validation'],test=evaluated['metrics'],
        holdout_isolation_checked=evaluated['holdout_isolation_checked'],
        test_prediction_rows=rows,preparation=prep,seconds=time.monotonic()-started,
        checkpoint_sha256=file_hash(run/'model/best.pt'))
    with (run/'test_predictions.csv').open('rb') as src,gzip.open(run/'test_predictions.csv.gz','wb',compresslevel=6) as dst:
        shutil.copyfileobj(src,dst)
    (run/'test_predictions.csv').unlink()
    # These are newly built reproducible temporary inputs, never original source data.
    shutil.rmtree(inputs)
    atomic_json(run/'result.json',result)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['qualify','smoke','batch'],required=True)
    p.add_argument('--output',default=os.environ.get('CQ_OUTPUT_DIR'))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--train-cap',type=int,default=500000)
    p.add_argument('--validation-cap',type=int,default=100000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--datasets',nargs='+',default=DATASETS)
    args=p.parse_args()
    if not args.output:raise ValueError('output directory is required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(out/'.hf_cache'))
    if args.stage=='qualify':
        subprocess.run([sys.executable,'-m','pytest','tests','-q'],check=True)
        subprocess.run([sys.executable,'-m','scripts.data.transaction_data','verify','--source-bytes'],check=True)
        atomic_json(out/'summary.json',dict(status='complete',stage='qualify',torch=str(torch.__version__),cuda_available=torch.cuda.is_available()))
        (out/'progress.txt').write_text('1/1 qualification complete\n')
        return
    datasets=['sd-crypto'] if args.stage=='smoke' else args.datasets
    if args.stage=='smoke':args.epochs=1
    protocol=dict(stage=args.stage,datasets=datasets,seed=args.seed,epochs=args.epochs,
        train_cap=args.train_cap,validation_cap=args.validation_cap,device=args.device,
        protocol='fixed single-seed short-budget run; complete temporal holdout; no test-based tuning',
        private_ieee_cis='excluded from cloud; evaluate locally only')
    atomic_json(out/'protocol.json',protocol)
    results=[]
    for index,dataset in enumerate(datasets):
        with (out/'progress.txt').open('a') as f:f.write(f'{index}/{len(datasets)} starting {dataset}\n')
        try:
            result=run_dataset(dataset,args,out)
        except Exception as error:
            traceback.print_exc()
            result=dict(dataset=dataset,status='failed',error=f'{type(error).__name__}: {error}')
        results.append(result)
        atomic_json(out/'summary.json',dict(status='complete' if all(r['status']=='complete' for r in results) and len(results)==len(datasets) else 'in_progress',
            protocol=protocol,results=results,
            peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.,
            peak_gpu_reserved_gb=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0.))
        with (out/'progress.txt').open('a') as f:f.write(f'{index+1}/{len(datasets)} {dataset} {result["status"]}\n')
    if (out/'.hf_cache').exists():shutil.rmtree(out/'.hf_cache')
    if any(r['status']!='complete' for r in results):raise RuntimeError('some datasets failed; see summary.json')


if __name__=='__main__':main()
