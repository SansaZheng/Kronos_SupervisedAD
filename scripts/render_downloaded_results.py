"""Render aggregate-only benchmark figures, table and a self-contained HTML report.

No transaction-level records are copied to the report output. Missing/failed runs
stay visible and are never filled with older experiments or invented metrics.
"""
import argparse
import base64
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER=['sd-stock','sd-crypto','tstr','sapienza','ulb','elliptic','paysim','ibm_aml','tabformer','ieee_cis']
NAMES={'sd-stock':'SD-FMM · stock','sd-crypto':'SD-FMM · crypto','tstr':'TSTR','sapienza':'Sapienza · 15S',
       'ulb':'ULB Credit Card','elliptic':'Elliptic','paysim':'PaySim','ibm_aml':'IBM AML · HI-Small',
       'tabformer':'IBM TabFormer','ieee_cis':'IEEE-CIS · local'}


def read_results(public, private):
    records={}; curves={}
    for directory in public + private:
        directory=Path(directory)
        summary=directory/'summary.json'
        entries=json.loads(summary.read_text()).get('results',[]) if summary.exists() else []
        if (directory/'result.json').exists(): entries+=[json.loads((directory/'result.json').read_text())]
        for result_file in directory.glob('*/result.json'):
            entries+=[json.loads(result_file.read_text())]
        for record in entries:
            name=record['dataset']
            if name=='ieee_cis':
                # Category vocabulary contains source values; export only its size.
                record=json.loads(json.dumps(record))
                prep=record.get('preparation',{})
                names=prep.pop('feature_names',[])
                if names:prep['feature_count']=len(names)
            if name in records and records[name]!=record:
                raise ValueError(f'conflicting results for {name}; choose exactly one run')
            records[name]=record
        for curve_file in [directory/'pr_curve.json',*directory.glob('*/pr_curve.json')]:
            if curve_file.exists():
                name=directory.name if curve_file.parent==directory else curve_file.parent.name
                if name not in ORDER and (curve_file.parent/'result.json').exists():
                    name=json.loads((curve_file.parent/'result.json').read_text())['dataset']
                curves[name]=json.loads(curve_file.read_text())
    # Rebuild display curves from complete saved scores. Uniform subsampling by
    # threshold index can erase the rare-positive, high-precision part of a curve.
    import pandas as pd
    from sklearn.metrics import precision_recall_curve
    for name,record in records.items():
        if record['status']!='complete':continue
        candidates=[p for d in public+private for p in (Path(d)/name/'test_predictions.csv.gz',Path(d)/'test_predictions.csv.gz')]
        prediction=next((p for p in candidates if p.exists() and
            json.loads((p.parent/'result.json').read_text())['dataset']==name),None)
        if prediction is None:continue
        table=pd.read_csv(prediction,usecols=['label','anomaly_probability'],float_precision='round_trip')
        known=table.label>=0
        precision,recall,thresholds=precision_recall_curve(table.loc[known,'label'],table.loc[known,'anomaly_probability'],drop_intermediate=True)
        bins=np.floor(recall*1000).astype(int)
        edges=np.flatnonzero(bins[1:]!=bins[:-1])
        operating=min(int(np.searchsorted(thresholds,record['test']['threshold'])),len(precision)-1)
        selected=np.unique(np.r_[0,edges,edges+1,len(precision)-1,operating])
        curves[name]=dict(precision=precision[selected].tolist(),recall=recall[selected].tolist(),prevalence=record['test']['prevalence'])
    return records,curves


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public',nargs='*',default=[])
    p.add_argument('--private',nargs='*',default=[])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    records,curves=read_results(args.public,args.private)
    if not records:raise ValueError('no results found; refuse to make an empty benchmark report')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    complete=[records[n] for n in ORDER if n in records and records[n]['status']=='complete']
    available=[r for r in complete if r['test'].get('average_precision') is not None]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    figures=[]
    if available:
        fig,axes=plt.subplots(1,2,figsize=(13,max(4,len(available)*.46+1.8)),sharey=True)
        y=np.arange(len(available)); names=[NAMES[r['dataset']] for r in available]
        axes[0].barh(y,[r['test']['average_precision'] for r in available],color='#1d718d',height=.57,label='Test AP')
        axes[0].scatter([r['test']['prevalence'] for r in available],y,color='#db7b2b',s=32,zorder=3,label='Positive prevalence')
        axes[1].barh(y,[r['test']['f1'] for r in available],color='#389b83',height=.57)
        axes[0].set_yticks(y,names);axes[0].invert_yaxis()
        for ax,key,title in [(axes[0],'average_precision','Average precision (AP)'),(axes[1],'f1','F1 at validation-selected threshold')]:
            ax.set_xlim(0,1.12);ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
            for index,r in enumerate(available):
                value=r['test'][key];ax.text(value+.015,index,f'{value:.4f}',va='center',fontsize=9)
            ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Test score')
        axes[0].legend(loc='lower right',frameon=False,fontsize=8)
        fig.suptitle('Kronos SupervisedAD · downloaded-data holdouts',x=.035,ha='left',fontsize=17,fontweight='bold')
        fig.text(.035,.015,'Single seed 42 · up to 5 epochs · independent validation selects epoch and threshold · units differ across datasets',fontsize=9,color='#586470')
        fig.tight_layout(rect=(0,.04,1,.93))
        fig.savefig(out/'scores.png',dpi=180);fig.savefig(out/'scores.svg');plt.close(fig);figures.append(('scores.png','各数据集测试 AP 与 F1。橙点为测试集异常占比。'))
    present=[n for n in ORDER if n in curves and n in records and records[n]['status']=='complete']
    if present:
        columns=3; rows=(len(present)+columns-1)//columns
        fig,axes=plt.subplots(rows,columns,figsize=(13,3.3*rows),squeeze=False)
        for ax,n in zip(axes.flat,present):
            curve=curves[n];metric=records[n]['test']
            ax.step(curve['recall'],curve['precision'],where='post',color='#1d718d',lw=1.6)
            ax.axhline(curve['prevalence'],color='#db7b2b',ls='--',lw=1)
            ax.scatter([metric['recall']],[metric['precision']],s=35,color='#c84f51',zorder=4)
            ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Recall',ylabel='Precision',title=f"{NAMES[n]} | AP {metric['average_precision']:.4f}")
            ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(present):]:ax.set_visible(False)
        fig.suptitle('Precision–recall curves · red dot = fixed operating threshold',fontsize=16,fontweight='bold')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'precision_recall.png',dpi=160);plt.close(fig)
        figures.append(('precision_recall.png','精确率–召回率曲线；红点是验证集预先选定阈值的实际测试表现，虚线是异常占比。'))
    # Deterministic illustrations, never examples selected by model score/performance.
    examples=[]
    for name in ('sd-stock','sd-crypto'):
        candidates=[Path(d)/name/'test_predictions.csv.gz' for d in args.public]
        predictions=next((path for path in candidates if path.exists()),None)
        if predictions is None or name not in records or records[name]['status']!='complete':continue
        import pandas as pd
        frame=pd.read_csv(predictions)
        positive=frame.loc[frame.label==1,'sample_id']
        if not len(positive):continue
        sample=frame.loc[frame.sample_id==positive.iloc[0]].sort_values('step')
        examples.append((name,sample,records[name]['test']['threshold']))
    if examples:
        fig,axes=plt.subplots(len(examples),1,figsize=(13,3.4*len(examples)),squeeze=False)
        for ax,(name,sample,threshold) in zip(axes.flat,examples):
            ax.plot(sample.step,sample.anomaly_probability,color='#1d718d',lw=1.6,label='Anomaly score')
            ax.axhline(threshold,color='#a87623',ls='--',lw=1.3,label='Fixed threshold')
            for step in sample.loc[sample.label==1,'step']:ax.axvspan(step-.45,step+.45,color='#c84f51',alpha=.3)
            alerts=sample.prediction==1
            ax.scatter(sample.loc[alerts,'step'],sample.loc[alerts,'anomaly_probability'],color='#9a4f96',s=16,zorder=3,label='Predicted alert')
            ax.set(xlabel='Time point within sequence',ylabel='Score (sigmoid)',ylim=(0,1.03),title=f'{NAMES[name]} | first test sequence containing a labelled anomaly')
            ax.grid(alpha=.12);ax.legend(loc='upper right',fontsize=8)
        fig.suptitle('Detection examples · red shading = labelled anomaly points',fontsize=16,fontweight='bold')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'detection_examples.png',dpi=170);plt.close(fig)
        figures.append(('detection_examples.png','实际检测示例：固定选取各测试文件中首个含标签异常的序列，不依据模型表现挑选。红色区域为真值，紫点为模型报警。'))
    columns=['dataset','status','unit','test_n','positives','prevalence','AP','ROC_AUC','precision','recall','F1','FPR','TP','FP','FN','TN','threshold','epochs','best_epoch']
    rows=[]
    for n in ORDER:
        r=records.get(n,{'dataset':n,'status':'not_run'})
        if r['status']=='complete':
            m=r['test'];row=[NAMES[n],r['status'],r['unit'],m['n'],m['positives'],m['prevalence'],m['average_precision'],m['roc_auc'],m['precision'],m['recall'],m['f1'],m['false_positive_rate'],m['tp'],m['fp'],m['fn'],m['tn'],m['threshold'],r['epochs_executed'],r['best_epoch']]
        else:row=[NAMES[n],r['status']]+['']*(len(columns)-2)
        rows.append(row)
    import csv
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
    safe_records={n:r for n,r in records.items()}
    (out/'aggregate_results.json').write_text(json.dumps(safe_records,indent=2,allow_nan=False)+'\n')
    heads=['数据集','状态','检测单位','测试数','异常数','异常占比','AP','ROC-AUC','精确率','召回率','F1','FPR','TP','FP','FN','TN','阈值','训练轮数','最佳轮数']
    def cell(v):
        return f'{v:.4f}' if isinstance(v,float) else str(v)
    table='<thead><tr>'+''.join(f'<th>{h}</th>' for h in heads)+'</tr></thead><tbody>'
    table+=''.join('<tr>'+''.join('<td>'+html.escape(cell(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody>'
    image_html=''
    for filename,caption in figures:
        encoded=base64.b64encode((out/filename).read_bytes()).decode()
        image_html+=f'<figure><img alt="{caption}" src="data:image/png;base64,{encoded}"><figcaption>{caption}</figcaption></figure>'
    notes='''<ul><li>每项数据独立训练同一套监督式 Kronos 方法。预训练 Kronos-mini + 特征适配器 + 多尺度异常头；类别加权 BCE。</li>
<li>固定随机种子 42，最多 5 轮，验证集 AP 选择模型，验证集 F1 选择阈值；测试集只用于最终评估。本结果为短预算初测，不能代表充分调参后的性能。</li>
<li>PaySim、TabFormer、IBM AML 的训练目标最多 500,000 条、验证目标最多 100,000 条，按与标签无关的整块哈希选择；测试段保留全部交易。上下文使用原始的历史记录。</li>
<li>ULB、上述三项大交易数据、IEEE-CIS 使用时间划分；Elliptic 按时间步 1–34 / 35–41 / 42–49 划分；其余沿用仓库可复现的研究划分，不宣称官方零样本结果。市场序列是发布者加工的研究片段，不据此宣称实时交易效果；Sapienza 采用 15S，5S／25S 是同一批事件的其他分辨率。</li>
<li>Elliptic 为交易节点特征分类，未使用图边，未知标签不参与指标。TSTR 按整段序列判异常；Sapienza 和 SD-FMM 按时间点判异常，不将不同检测单位的分数求总平均。</li>
<li>IEEE-CIS 仅使用官方有标签训练文件的时间留出集，官方无标签测试文件不作真实标签。原始交易、派生输入及逐笔预测仅在本机保存。</li>
<li>类别加权 BCE 的 sigmoid 异常分数未经概率校准，不能直接解释为真实欺诈风险概率。</li>
<li>AP 是 average precision，不是梯形积分 PR-AUC。AP 与异常占比共同阅读；高准确率不能说明稀有异常检测有效。阈值在验证集上选定，因此测试 F1 可能显著下降。</li></ul>'''
    observations=[]
    if 'tstr' in records and records['tstr']['status']=='complete':
        m=records['tstr']['test']; baseline=2*m['prevalence']/(1+m['prevalence'])
        observations.append(f"TSTR：AP={m['average_precision']:.4f}，异常占比={m['prevalence']:.4f}；固定阈值 F1={m['f1']:.4f}，而全报异常的 F1={baseline:.4f}。这项数据不能只凭 F1 较高判断有效。")
    for name in ('paysim','tabformer'):
        if name in records and records[name]['status']=='complete':
            m=records[name]['test']
            observations.append(f"{NAMES[name]}：检测出 {m['tp']:,}/{m['positives']:,} 个异常，误报 {m['fp']:,} 条；召回率 {100*m['recall']:.2f}%。这是真实保留的低召回结果，未利用测试集重新调阈值。")
    observation_html='<section class="card"><h2>本批次观察</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in observations)+'</ul></section>' if observations else ''
    details=''
    for r in complete:
        prep=r.get('preparation',{})
        details+=f'<details><summary>{html.escape(NAMES[r["dataset"]])} · 数据覆盖与限制</summary><pre>'+html.escape(json.dumps(prep,indent=2,ensure_ascii=False))+'</pre></details>'
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kronos SupervisedAD 实测结果</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;color:#243443;background:#f5f7fa;margin:0}}main{{max-width:1400px;padding:36px;margin:auto}}h1{{font-size:30px}}.card,figure,details{{background:white;border-radius:12px;padding:24px;margin:20px 0}}img{{max-width:100%;height:auto}}figcaption{{color:#576471}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px 12px;border-bottom:1px solid #e3e8ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead{{background:#eef4f7}}pre{{white-space:pre-wrap;font-size:12px}}input{{padding:10px;width:300px;max-width:90%;border:1px solid #c4d1da;border-radius:6px}}.muted{{color:#627181}}</style>
<main><h1>Kronos SupervisedAD · 下载数据实测</h1><p class="muted">已完成 {len(complete)} / {len(ORDER)} 项。下表仅填入本批次真实输出；失败或未运行的数据保留状态。</p>{image_html}{observation_html}<section class="card"><h2>测试集指标</h2><input id="filter" placeholder="筛选数据集" aria-label="筛选数据集"><div class="table"><table id="metrics">{table}</table></div></section><section class="card"><h2>如何解读</h2>{notes}</section>{details}<p class="muted">本报告只包含汇总指标与曲线，不包含交易级记录。机器可读结果见同目录 metrics.csv 与 aggregate_results.json。</p></main><script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('#metrics tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}})</script></html>'''
    (out/'report.html').write_text(report)
    print(json.dumps({'output':str(out),'complete':len(complete),'datasets':[r['dataset'] for r in complete]}))


if __name__=='__main__':main()
