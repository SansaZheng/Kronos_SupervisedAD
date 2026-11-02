"""Local-only IEEE-CIS labelled-train holdout; never upload source/derived rows."""
import argparse
import gc
import gzip
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from supervised.data import chunk_point_stream,save,check_splits,load
from supervised.__main__ import parser
from supervised.engine import train,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',required=True)
    p.add_argument('--backbone',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=5)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    numeric=['TransactionDT','TransactionAmt','dist1','dist2']+[f'C{i}' for i in range(1,15)]
    categorical=['ProductCD']+[f'card{i}' for i in range(1,7)]
    usecols=['TransactionID','isFraud']+numeric+categorical
    with zipfile.ZipFile(args.zip) as archive:
        with archive.open('train_transaction.csv') as stream:
            frame=pd.read_csv(stream,usecols=usecols,low_memory=False)
    frame=frame.sort_values(['TransactionDT','TransactionID'],kind='stable').reset_index(drop=True)
    cuts=[0]+[int(frame.TransactionDT.searchsorted(frame.TransactionDT.iloc[int(len(frame)*f)],side='left'))
              for f in (.7,.85)]+[len(frame)]
    encoder=OneHotEncoder(max_categories=16,handle_unknown='infrequent_if_exist',sparse_output=False,dtype=np.float32)
    cat=frame[categorical].fillna('__missing__').astype(str)
    encoder.fit(cat.iloc[:cuts[1]])
    frame['TransactionDT']=frame.TransactionDT/86400
    frame['TransactionAmt']=np.log1p(frame.TransactionAmt)
    names=['relative_day','log_transaction_amount','dist1','dist2']+[f'count_{i}' for i in range(1,15)]
    names+=['category_'+s for s in encoder.get_feature_names_out()]
    data=out/'inputs';data.mkdir();summaries={}
    for i,name in enumerate(['train','validation','test']):
        rows=frame.iloc[cuts[i]:cuts[i+1]]
        x=np.column_stack([rows[numeric].to_numpy(dtype=np.float32),encoder.transform(cat.iloc[cuts[i]:cuts[i+1]])])
        split=chunk_point_stream(x,rows.isFraud.to_numpy(dtype=np.int64),
            [f'ieee:{i}' for i in rows.TransactionID],128,32,'ieee:'+name,names)
        save(data/f'{name}.npz',split)
        summaries[name]=dict(rows=len(rows),positives=int(rows.isFraud.sum()),sampled=False)
        del x,split
    del frame,cat
    gc.collect()
    check_splits([load(data/f'{s}.npz') for s in ('train','validation','test')])
    preparation=dict(dataset='ieee_cis',summaries=summaries,feature_names=names,
        protocol='70/15/15 chronological split of official labelled train; original official test is not used as truth',
        feature_subset='transaction time/amount/distance/count fields and card/product categories; no identity join or V features',
        limits='anonymized C feature availability at decision time is not independently verified',
        raw_and_derived_data_location='local private output only; never GitHub or cloud jobs')
    (out/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    train_args=parser().parse_args(['train','--data-dir',str(data),'--backbone',args.backbone,
        '--local-only','--epochs',str(args.epochs),'--batch-size','32','--device','cpu',
        '--defer-test','--output',str(out/'model')])
    report=train(train_args)
    predict(argparse.Namespace(command='evaluate',checkpoint=str(out/'model/best.pt'),
        data=str(data/'test.npz'),output=str(out/'test_predictions.csv'),device='cpu',batch_size=32,threads=2))
    evaluated=json.loads((out/'test_predictions.csv.metrics.json').read_text())
    result=dict(dataset='ieee_cis',status='complete',unit='transaction',validation=report['validation'],
        test=evaluated['metrics'],epochs_executed=report['epochs_executed'],best_epoch=report['best_epoch'],
        preparation=preparation,holdout_isolation_checked=evaluated['holdout_isolation_checked'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'test_predictions.csv').open('rb') as source,gzip.open(out/'test_predictions.csv.gz','wb') as target:
        shutil.copyfileobj(source,target)
    (out/'test_predictions.csv').unlink()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
