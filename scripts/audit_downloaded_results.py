"""Independently recompute aggregate metrics from saved held-out predictions.

Writes hashes and aggregate differences only, never transaction records.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,roc_auc_score,
    confusion_matrix,precision_score,recall_score,f1_score,brier_score_loss)


def audit(directory):
    directory=Path(directory)
    record=json.loads((directory/'result.json').read_text())
    expected=record['test'];threshold=expected['threshold']
    prediction_path=directory/'test_predictions.csv.gz'
    table=pd.read_csv(prediction_path,usecols=['label','anomaly_probability','prediction'],float_precision='round_trip')
    assert np.isfinite(table.anomaly_probability).all()
    assert table.anomaly_probability.between(0,1).all()
    assert table.label.isin([-1,0,1]).all()
    assert np.array_equal((table.anomaly_probability>=threshold).astype(int),table.prediction)
    known=table.label>=0;y=table.loc[known,'label'];p=table.loc[known,'anomaly_probability'];a=table.loc[known,'prediction']
    tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1]).ravel()
    actual=dict(n=len(y),positives=int(y.sum()),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),
        precision=precision_score(y,a,zero_division=0),recall=recall_score(y,a,zero_division=0),
        f1=f1_score(y,a,zero_division=0),false_positive_rate=fp/max(fp+tn,1),
        false_alerts_per_1000=1000*fp/len(y),prevalence=float(y.mean()),
        brier_score=brier_score_loss(y,p),
        average_precision=average_precision_score(y,p) if len(set(y))==2 else None,
        roc_auc=roc_auc_score(y,p) if len(set(y))==2 else None)
    differences={}
    for key,value in actual.items():
        if value is None:
            assert expected[key] is None,key
        else:
            delta=abs(float(expected[key])-float(value))
            if delta>1e-10:differences[key]=delta
    assert not differences,f'metrics disagree for {directory.name}: {differences}'
    assert record['holdout_isolation_checked'] is True
    digest=hashlib.sha256()
    with prediction_path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return dict(dataset=record['dataset'],status='verified',prediction_rows=len(table),
        known_labels=len(y),unknown_labels=int((~known).sum()),prediction_sha256=digest.hexdigest(),
        metrics_recomputed=list(actual),maximum_allowed_difference=1e-10,
        probabilities_valid=True,alerts_match_fixed_threshold=True,holdout_isolation_checked=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+')
    p.add_argument('--output',required=True)
    args=p.parse_args();results=[]
    for root in args.directories:
        root=Path(root)
        directories=[root] if (root/'result.json').exists() else sorted(f.parent for f in root.glob('*/result.json'))
        for directory in directories:
            result=audit(directory);results.append(result);print(result['dataset'],result['status'],flush=True)
    if not results:raise ValueError('no completed result files to audit')
    Path(args.output).write_text(json.dumps(dict(status='verified',results=results),indent=2)+'\n')


if __name__=='__main__':main()
