import numpy as np
import pandas as pd

from supervised.data import load
from supervised.transaction_prepare import prepare_transactions, frame_features, priority


def test_label_blind_sampling_keeps_full_chronological_test_and_real_context(tmp_path,monkeypatch):
    import supervised.transaction_prepare as prep
    frame=pd.DataFrame(dict(step=np.arange(1,101),type=['TRANSFER']*100,
        amount=np.arange(100)+1.,oldbalanceOrg=np.arange(100)+100.,oldbalanceDest=np.arange(100)+50.,
        isFraud=(np.arange(100)%7==0).astype(int)))
    def chunks(dataset,root):
        for start in range(0,len(frame),17):yield frame.iloc[start:start+17].reset_index(drop=True)
    monkeypatch.setattr(prep,'source_chunks',chunks)
    report=prepare_transactions('paysim',tmp_path,tmp_path/'first',train_cap=16,validation_cap=8,
                                 chunk_length=4,history_length=2)
    train,validation,test=[load(tmp_path/f'first/{s}.npz') for s in ('train','validation','test')]
    assert test.target_mask.sum()==report['summaries']['test']['source_rows']
    assert train.target_mask.sum()<=16 and validation.target_mask.sum()<=8
    assert np.any(train.target_mask.sum(axis=1)<train.lengths)
    assert not (tmp_path/'first/.scratch').exists()
    frame.loc[frame.step>=85,'isFraud']=1-frame.loc[frame.step>=85,'isFraud']
    prepare_transactions('paysim',tmp_path,tmp_path/'second',train_cap=16,validation_cap=8,
                           chunk_length=4,history_length=2)
    for name in ('train','validation'):
        a,b=load(tmp_path/f'first/{name}.npz'),load(tmp_path/f'second/{name}.npz')
        np.testing.assert_array_equal(a.X,b.X)
        np.testing.assert_array_equal(a.y,b.y)
        np.testing.assert_array_equal(a.observation_id,b.observation_id)


def test_tabformer_refunds_and_timestamp_fields_are_not_target_features():
    frame=pd.DataFrame({'Year':[2019,2019],'Month':[1,1],'Day':[1,2],'Time':['12:30','00:10'],
        'Amount':['$10.00','$-2.50'],'Use Chip':['Swipe Transaction','Online Transaction'],
        'Merchant State':['CA',None],'MCC':[1234,2345],'Is Fraud?':['No','Yes']})
    numeric,categories,y,time,period=frame_features('tabformer',frame)
    assert numeric.log_amount.iloc[1]<0 and y.tolist()==[0,1]
    assert 'Is Fraud?' not in categories and 'Is Fraud?' not in numeric
    assert time[1]>time[0] and period[1]>period[0]
    np.testing.assert_array_equal(priority(np.arange(10)),priority(np.arange(10)))
