"""Predeclared validation-only repair rounds, then sealed historical-holdout scoring."""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supervised.__main__ import parser
from supervised.data import load,prepare_builtin
from supervised.engine import train,predict,file_hash
from supervised.transaction_prepare import prepare_transactions,prepare_elliptic
from scripts.run_downloaded_benchmark import DATASETS,atomic_json

# Fixed before running any new validation/test results. R1 isolates the head;
# R2/R3 are budget/optimization bundles, not clean single-factor ablations.
ROUNDS=[
 dict(name='r0_reference',head='multiscale',epochs=5,lr=1e-4,ratio=.1,cap=None),
 dict(name='r1_raw_residual',head='hybrid',epochs=5,lr=1e-4,ratio=.1,cap=None),
 dict(name='r2_faster_longer',head='hybrid',epochs=12,lr=1e-3,ratio=.1,cap=None),
 dict(name='r3_capped_weight',head='hybrid',epochs=12,lr=3e-4,ratio=.3,cap=100.),
]


def training(inputs,out,dataset,config,seed,device,backbone):
    command=['train','--data-dir',str(inputs),'--output',str(out),'--backbone',backbone,
      '--epochs',str(config['epochs']),'--lr',str(config['lr']),
      '--backbone-lr-ratio',str(config['ratio']),'--head-type',config['head'],
      '--seed',str(seed),'--batch-size','4' if dataset=='tstr' else '32',
      '--threads','2','--device',device,'--defer-test','--patience','3']
    if dataset=='tstr':command+=['--max-context','2048']
    if config['cap'] is not None:command+=['--max-pos-weight',str(config['cap'])]
    return train(parser().parse_args(command))


def compress(path):
    if not path.exists():return
    with path.open('rb') as source,gzip.open(str(path)+'.gz','wb',compresslevel=6) as target:shutil.copyfileobj(source,target)
    path.unlink()


def evaluate(model,inputs,output,dataset,device):
    predict(argparse.Namespace(command='evaluate',checkpoint=str(model/'best.pt'),data=str(inputs/'test.npz'),
      output=str(output),device=device,batch_size=4 if dataset=='tstr' else 32,threads=2))
    return json.loads(Path(str(output)+'.metrics.json').read_text())


def run_dataset(dataset,args,out):
    started=time.monotonic();run=out/dataset;run.mkdir(parents=True,exist_ok=False)
    inputs=run/'inputs';root=Path('data/supervised_financial')
    if args.prepared:
        shutil.copytree(args.prepared,inputs)
    elif dataset in ('paysim','ibm_aml','tabformer'):
        prepare_transactions(dataset,root,inputs,500000,100000,seed=42)
    elif dataset=='elliptic':prepare_elliptic(root,inputs)
    else:prepare_builtin(dataset,root,inputs,history_length=32 if dataset in ('ulb','sapienza') else 0)
    prep=json.loads((inputs/'preparation.json').read_text()) if (inputs/'preparation.json').exists() else {}
    # The training entrypoint cannot accidentally load a test.npz in trial runs.
    (inputs/'test.npz').rename(inputs/'sealed_test.npz')
    trials=[];configs=ROUNDS[:2] if args.stage=='smoke' else ROUNDS
    for config in configs:
        model=run/config['name']
        if args.stage=='smoke':config={**config,'epochs':1}
        report=training(inputs,model,dataset,config,42,args.device,args.backbone)
        assert report['test_deferred'] and report['test'] is None
        trials.append(dict(config=config,validation=report['validation'],epochs_executed=report['epochs_executed'],
            best_epoch=report['best_epoch'],checkpoint_sha256=file_hash(model/'best.pt')))
        compress(model/'validation_predictions.csv')
        with (out/'progress.txt').open('a') as f:f.write(f'{dataset} {config["name"]} validation complete\n')
    # Exact ties favor the less complex earlier candidate. No test values enter selection.
    best=max(range(len(trials)),key=lambda i:(trials[i]['validation']['average_precision'],trials[i]['validation']['f1'],-i))
    selection=dict(dataset=dataset,selected=trials[best]['config'],selected_index=best,trials=trials,
      policy='validation AP, then validation F1, then earlier candidate; reference is eligible',
      test_file_absent_during_training=not (inputs/'test.npz').exists(),historical_test_already_reported=True)
    atomic_json(run/'validation_selection.json',selection)
    seal=file_hash(run/'validation_selection.json')
    chosen=trials[best]['config'];model=run/chosen['name']
    (inputs/'sealed_test.npz').rename(inputs/'test.npz')
    scored=evaluate(model,inputs,run/'test_predictions.csv',dataset,args.device)
    unit='whole sequence' if dataset=='tstr' else 'transaction node' if dataset=='elliptic' else 'time point' if dataset in ('sd-stock','sd-crypto','sapienza') else 'transaction'
    result=dict(dataset=dataset,status='complete',unit=unit,seed=42,
      epochs_executed=trials[best]['epochs_executed'],best_epoch=trials[best]['best_epoch'],
      validation=trials[best]['validation'],test=scored['metrics'],
      holdout_isolation_checked=scored['holdout_isolation_checked'],preparation=prep,
      checkpoint_sha256=scored['checkpoint_sha256'],selected_round=chosen['name'],
      validation_selection_sha256=seal,selection_test_independent=True,seconds=time.monotonic()-started)
    compress(run/'test_predictions.csv')
    # Extra seeds check selected recipe vs reference; choices never change afterward.
    confirmations=[]
    if args.stage=='batch' and dataset in ('ulb','paysim','tabformer') and best!=0:
        for seed in (13,2026):
            for config in (ROUNDS[0],chosen):
                directory=run/f'confirmation_{seed}_{config["name"]}'
                report=training(inputs,directory,dataset,config,seed,args.device,args.backbone)
                prediction=run/f'confirmation_{seed}_{config["name"]}.csv'
                evaluated=evaluate(directory,inputs,prediction,dataset,args.device)
                confirmations.append(dict(seed=seed,config=config,validation=report['validation'],
                    test=evaluated['metrics'],checkpoint_sha256=evaluated['checkpoint_sha256'],
                    holdout_isolation_checked=evaluated['holdout_isolation_checked']))
                prediction.unlink() # newly generated confirmation rows; aggregate metrics retained
                (directory/'validation_predictions.csv').unlink()
                (directory/'best.pt').unlink()
        atomic_json(run/'confirmation.json',dict(selection_frozen=True,runs=confirmations))
    assert file_hash(run/'validation_selection.json')==seal
    for trial in trials:
        directory=run/trial['config']['name']
        if directory!=model:(directory/'best.pt').unlink()
    model.rename(run/'model')
    shutil.rmtree(inputs) # only freshly prepared copies, never original source assets
    atomic_json(run/'result.json',result)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','batch','local'],required=True)
    p.add_argument('--output',default=os.environ.get('CQ_OUTPUT_DIR'))
    p.add_argument('--datasets',nargs='+',default=DATASETS)
    p.add_argument('--device',default='cuda')
    p.add_argument('--backbone',default='NeoQuasar/Kronos-mini')
    p.add_argument('--prepared',help='local-only prepared private inputs; never use in cloud specs')
    args=p.parse_args()
    if not args.output:raise ValueError('output is required')
    if args.prepared and (args.stage!='local' or len(args.datasets)!=1):raise ValueError('private prepared data are local-only and single-dataset')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(rounds=ROUNDS,seed=42,confirmation_seeds=[13,2026],confirmation_datasets=['ulb','paysim','tabformer'],
      selection='per-dataset validation AP/F1 only; baseline eligible; no test-driven retries or threshold changes',
      historical_holdout_note='previous test results have already been seen; this is a historical-holdout reassessment, not a new untouched external test',
      datasets=['sd-crypto'] if args.stage=='smoke' else args.datasets)
    atomic_json(out/'protocol.json',protocol)
    results=[]
    for dataset in protocol['datasets']:
        (out/'progress.txt').open('a').write(f'{len(results)}/{len(protocol["datasets"])} starting {dataset}\n')
        results.append(run_dataset(dataset,args,out))
        atomic_json(out/'summary.json',dict(status='complete' if len(results)==len(protocol['datasets']) else 'in_progress',
            protocol=protocol,results=results,peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0))
        (out/'progress.txt').open('a').write(f'{len(results)}/{len(protocol["datasets"])} complete {dataset}\n')


if __name__=='__main__':main()
