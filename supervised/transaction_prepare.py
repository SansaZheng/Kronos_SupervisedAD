"""Label-independent chronological preparation of the downloaded transaction assets.

Large sources are scanned in chunks. A capped training/validation sample is
selected by row-ID hash, never by label. The entire temporal test segment is kept.
"""
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from scripts.data.transaction_data import iter_table, map_labels
from .data import Split, chunk_point_stream, save, load, check_splits


FIELDS = {
 'paysim': ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud'],
 'tabformer': ['Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant State', 'MCC', 'Is Fraud?'],
 'ibm_aml': ['Timestamp', 'From Bank', 'To Bank', 'Amount Received', 'Receiving Currency',
             'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering'],
}


def signed_log(values):
    x = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def frame_features(dataset, frame):
    if dataset == 'paysim':
        time = pd.to_numeric(frame.step).to_numpy(dtype=np.int64)
        numeric = pd.DataFrame({'relative_hour': time,
            'log_amount': signed_log(frame.amount),
            'log_origin_balance_before': signed_log(frame.oldbalanceOrg),
            'log_destination_balance_before': signed_log(frame.oldbalanceDest)})
        categorical = frame[['type']].astype(str).reset_index(drop=True)
        labels = map_labels('paysim', frame.isFraud)
        period = time
    elif dataset == 'tabformer':
        date = pd.to_datetime(dict(year=frame.Year, month=frame.Month, day=frame.Day))
        minute = pd.to_timedelta(frame.Time.astype(str) + ':00')
        stamp = date + minute
        time = stamp.astype('int64').to_numpy() // 10**9
        period = date.astype('int64').to_numpy() // (86400 * 10**9)
        hours = minute.dt.total_seconds().to_numpy() / 3600
        amount = frame.Amount.astype(str).str.replace('$', '', regex=False)
        numeric = pd.DataFrame({'relative_day': time / 86400,
            'hour_sin': np.sin(2*np.pi*hours/24), 'hour_cos': np.cos(2*np.pi*hours/24),
            'weekday_sin': np.sin(2*np.pi*date.dt.dayofweek.to_numpy()/7),
            'weekday_cos': np.cos(2*np.pi*date.dt.dayofweek.to_numpy()/7),
            'log_amount': signed_log(amount)})
        categorical = frame[['Use Chip','Merchant State','MCC']].fillna('__missing__').astype(str).reset_index(drop=True)
        labels = map_labels('tabformer', frame['Is Fraud?'])
    elif dataset == 'ibm_aml':
        stamp = pd.to_datetime(frame.Timestamp)
        time = stamp.astype('int64').to_numpy() // 10**9
        period = time // 3600
        hours = stamp.dt.hour.to_numpy() + stamp.dt.minute.to_numpy()/60
        numeric = pd.DataFrame({'relative_day': time / 86400,
            'hour_sin': np.sin(2*np.pi*hours/24), 'hour_cos': np.cos(2*np.pi*hours/24),
            'log_amount_paid': signed_log(frame['Amount Paid']),
            'log_amount_received': signed_log(frame['Amount Received']),
            'same_bank': (frame['From Bank'].astype(str).to_numpy() == frame['To Bank'].astype(str).to_numpy()).astype(float)})
        categorical = frame[['Receiving Currency','Payment Currency','Payment Format']].fillna('__missing__').astype(str).reset_index(drop=True)
        labels = map_labels('ibm_aml', frame['Is Laundering'])
    else:
        raise ValueError(dataset)
    return numeric.astype(np.float32), categorical, labels, np.asarray(time), np.asarray(period)


def source_chunks(dataset, root, chunksize=200_000):
    for frame in iter_table(dataset, chunksize=chunksize, root=Path(root)/'transactions',
                             usecols=FIELDS[dataset], low_memory=False):
        yield frame.reset_index(drop=True)


def priority(row_ids, seed=42):
    with np.errstate(over='ignore'):
        x = np.asarray(row_ids, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / (2.**53)


def prepare_transactions(dataset, source_root, output, train_cap=500_000, validation_cap=100_000,
                         chunk_length=128, history_length=32, seed=42):
    if dataset not in FIELDS or train_cap < 1 or validation_cap < 1:
        raise ValueError('invalid dataset or sample caps')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, classes, total = Counter(), Counter(), 0
    for frame in source_chunks(dataset, source_root):
        _, _, y, _, periods = frame_features(dataset, frame)
        counts.update(dict(zip(*np.unique(periods, return_counts=True))))
        classes.update(dict(zip(*np.unique(y, return_counts=True))))
        total += len(frame)
        print(f'{dataset} scan {total}', flush=True)
    keys = np.asarray(sorted(counts))
    cumulative = np.cumsum([counts[k] for k in keys])
    # Period ties stay together. Boundaries depend only on time and row counts.
    boundaries = [int(keys[min(int(np.searchsorted(cumulative, total*f)), len(keys)-1)]) for f in (.7,.85)]
    if boundaries[0] >= boundaries[1]:
        raise ValueError('not enough distinct time periods for temporal holdouts')
    full_counts = [int(sum(v for k,v in counts.items() if k < boundaries[0])),
                   int(sum(v for k,v in counts.items() if boundaries[0] <= k < boundaries[1])),
                   int(sum(v for k,v in counts.items() if k >= boundaries[1]))]
    selected = [[], []]
    offset = 0
    # Slight oversampling gives a deterministic hash reservoir; never balance by label.
    for frame in source_chunks(dataset, source_root):
        _, _, _, _, periods = frame_features(dataset, frame)
        ids = np.arange(offset, offset+len(frame)); offset += len(frame)
        assigned = np.searchsorted(boundaries, periods, side='right')
        scores = priority(ids, seed)
        for i, cap in enumerate((train_cap, validation_cap)):
            mask = (assigned == i) & (scores < min(1., 1.02*cap/max(full_counts[i],1)))
            kept = frame.loc[mask].copy()
            kept['_source_row'] = ids[mask]
            kept['_priority'] = scores[mask]
            selected[i].append(kept)
    samples = []
    for frames, cap in zip(selected, (train_cap, validation_cap)):
        sample = pd.concat(frames, ignore_index=True).nsmallest(cap, '_priority')
        numeric, categorical, y, time, _ = frame_features(dataset, sample)
        order = np.lexsort((sample._source_row.to_numpy(), time))
        samples.append((numeric.iloc[order], categorical.iloc[order], y[order], time[order],
                        sample._source_row.to_numpy()[order]))
    encoder = OneHotEncoder(max_categories=16, handle_unknown='infrequent_if_exist',
                            sparse_output=False, dtype=np.float32)
    encoder.fit(samples[0][1])
    feature_names = samples[0][0].columns.tolist() + ['category_'+s for s in encoder.get_feature_names_out()]
    # Encode the complete stream to temporary disk arrays, then sort row indices.
    # This keeps genuine past context without materializing 24M raw text rows.
    scratch=output/'.scratch';scratch.mkdir()
    feature_names += ['log_gap_seconds']
    raw_x=np.memmap(scratch/'x.bin',mode='w+',dtype=np.float32,shape=(total,len(feature_names)))
    raw_y=np.memmap(scratch/'y.bin',mode='w+',dtype=np.int8,shape=(total,))
    raw_time=np.memmap(scratch/'time.bin',mode='w+',dtype=np.int64,shape=(total,))
    raw_split=np.memmap(scratch/'split.bin',mode='w+',dtype=np.int8,shape=(total,))
    offset = 0
    for frame in source_chunks(dataset, source_root):
        numeric, categorical, y, time, periods = frame_features(dataset, frame)
        end=offset+len(frame)
        raw_x[offset:end,:-1]=np.column_stack([numeric.to_numpy(),encoder.transform(categorical)])
        raw_x[offset:end,-1]=0
        raw_y[offset:end]=y;raw_time[offset:end]=time
        raw_split[offset:end]=np.searchsorted(boundaries,periods,side='right')
        offset=end
        print(f'{dataset} encode {offset}/{total}',flush=True)
    raw_x.flush();raw_y.flush();raw_time.flush();raw_split.flush()
    del samples,selected
    summaries={}
    for split_index,name in enumerate(('train','validation','test')):
        indices=np.flatnonzero(raw_split==split_index)
        indices=indices[np.lexsort((indices,raw_time[indices]))]
        starts=np.arange(0,len(indices),chunk_length)
        cap=(train_cap,validation_cap,None)[split_index]
        if cap is not None and cap<len(indices):
            count=max(1,cap//chunk_length)
            chosen=np.argsort(priority(starts//chunk_length,seed+split_index),kind='stable')[:count]
            starts=np.sort(starts[chosen])
        width=chunk_length+history_length
        x=np.zeros((len(starts),width,len(feature_names)),dtype=np.float32)
        y=np.full((len(starts),width),-1,dtype=np.int64)
        targets=np.zeros((len(starts),width),dtype=bool)
        obs=np.full((len(starts),width),'',dtype=f'<U{len(dataset)+1+len(str(total))}')
        lengths=[];sample_ids=[]
        for i,start in enumerate(starts):
            begin,end=max(0,start-history_length),min(len(indices),start+chunk_length)
            rows=indices[begin:end];length=len(rows);target_offset=start-begin
            x[i,:length]=raw_x[rows]
            previous=np.r_[raw_time[indices[begin-1]] if begin else raw_time[rows[0]],raw_time[rows[:-1]]]
            x[i,:length,-1]=np.log1p(raw_time[rows]-previous)
            y[i,target_offset:length]=raw_y[indices[start:end]]
            targets[i,target_offset:length]=True
            obs[i,:length]=[f'{dataset}:{row}' for row in rows]
            lengths.append(length);sample_ids.append(f'{dataset}:{name}:{start}')
        split=Split(x,y,np.asarray(lengths),np.asarray(sample_ids),feature_names,
                    target_mask=targets,observation_id=obs)
        save(output/f'{name}.npz',split)
        known=split.valid_labels()
        summaries[name]=dict(rows=len(known),positives=int(known.sum()),source_rows=full_counts[split_index],
                             sampled=len(known)<full_counts[split_index],windows=len(x))
        del split,x,y,targets,obs,indices
    check_splits([load(output/f'{s}.npz') for s in ('train','validation','test')])
    del raw_x,raw_y,raw_time,raw_split
    shutil.rmtree(scratch)  # only these new temporary arrays; raw source assets remain intact
    report = dict(dataset=dataset,source_rows=total,source_class_counts={str(k):int(v) for k,v in classes.items()},
        split_boundaries=boundaries,period='step/hour' if dataset=='paysim' else 'day' if dataset=='tabformer' else 'hour',
        summaries=summaries, feature_names=feature_names, seed=seed, task='point',context_length=chunk_length+history_length,
        protocol='70/15/15 chronological period boundaries; label-independent block-hash caps for train/validation; complete test',
        limits='global stream, not account histories; stable source-row order breaks timestamp ties; categorical vocab fitted on label-independent training rows only',
        excluded_features='labels, rule flags, account/card/customer/merchant IDs, post-transaction balances, error annotations, laundering pattern annotations')
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def prepare_elliptic(source_root, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(source_root)/'transactions'
    features=pd.concat(iter_table('elliptic_features',root=root),ignore_index=True)
    classes=pd.concat(iter_table('elliptic_classes',root=root),ignore_index=True)
    label_map=dict(zip(classes.txId.astype(str),map_labels('elliptic',classes['class'])))
    ids=features.iloc[:,0].astype(str).to_numpy()
    time=features.iloc[:,1].to_numpy()
    y=np.asarray([label_map[i] for i in ids],dtype=np.int64)
    # Published 94 local features include the time step; omit all 72 aggregated-neighbor features.
    x=features.iloc[:,1:95].to_numpy(dtype=np.float32)
    names=['time_step']+[f'local_{i}' for i in range(1,94)]
    summaries={}
    for name,mask in [('train',time<=34),('validation',(time>=35)&(time<=41)),('test',time>=42)]:
        order=np.flatnonzero(mask);order=order[np.argsort(time[order],kind='stable')]
        split=Split(x[order,None,:],y[order,None],np.ones(len(order),dtype=np.int64),
                    np.asarray([f'elliptic:{i}' for i in ids[order]]),names,
                    observation_id=np.asarray([[f'elliptic:{i}'] for i in ids[order]]))
        save(output/f'{name}.npz',split)
        summaries[name]=dict(rows=len(order),positives=int((y[order]==1).sum()),
                             normals=int((y[order]==0).sum()),unknown=int((y[order]<0).sum()),sampled=False)
    check_splits([load(output/f'{s}.npz') for s in ('train','validation','test')])
    report=dict(dataset='elliptic',summaries=summaries,context_length=1,feature_names=names,
                protocol='time steps 1-34/35-41/42-49; unknown labels masked; local features only',
                limits='node-feature classification; edges not used; original entity-level labels are not judicial findings')
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
