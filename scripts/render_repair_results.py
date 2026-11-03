"""Compare all predeclared repair trials without hiding regressions or failed hypotheses."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.render_downloaded_results import read_results,ORDER,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public',nargs='*',default=[])
    p.add_argument('--private',nargs='*',default=[])
    p.add_argument('--reference',default='experiments/downloaded_20261103/aggregate_results.json')
    p.add_argument('--output',required=True)
    p.add_argument('--reference-public',nargs='*',default=[])
    p.add_argument('--reference-private',nargs='*',default=[])
    p.add_argument('--local-reference',help='optional same-MPS IEEE-CIS evaluate metrics JSON')
    args=p.parse_args();new,new_curves=read_results(args.public,args.private)
    old=json.loads(Path(args.reference).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    names=[n for n in ORDER if n in new and new[n]['status']=='complete']
    if not names:raise ValueError('no completed repairs to report')
    comparisons=[];trials={};confirmations={}
    for name in names:
        roots=[Path(d)/name for d in args.public+args.private]+[Path(d) for d in args.public+args.private]
        directory=next(r for r in roots if (r/'validation_selection.json').exists() and json.loads((r/'validation_selection.json').read_text())['dataset']==name)
        selection=json.loads((directory/'validation_selection.json').read_text());trials[name]=selection
        if (directory/'confirmation.json').exists():confirmations[name]=json.loads((directory/'confirmation.json').read_text())
        a,b=old[name]['test'],new[name]['test']
        comparisons.append(dict(dataset=name,selected_round=new[name]['selected_round'],unit=new[name]['unit'],
           n=b['n'],positives=b['positives'],old_AP=a['average_precision'],new_AP=b['average_precision'],
           delta_AP=b['average_precision']-a['average_precision'],old_F1=a['f1'],new_F1=b['f1'],delta_F1=b['f1']-a['f1'],
           old_precision=a['precision'],new_precision=b['precision'],old_recall=a['recall'],new_recall=b['recall'],
           old_FP=a['fp'],new_FP=b['fp'],new_TP=b['tp'],validation_selection_sha256=new[name]['validation_selection_sha256']))
    local_reference=json.loads(Path(args.local_reference).read_text()) if args.local_reference else None
    (out/'comparison.json').write_text(json.dumps(dict(comparisons=comparisons,results=new,validation_trials=trials,confirmation=confirmations,same_device_ieee_reference=local_reference),indent=2)+'\n')
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]),lineterminator='\n');writer.writeheader();writer.writerows(comparisons)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,max(4,len(names)*.55+1.6)),sharey=True);y=np.arange(len(names))
    for ax,key,title in [(axes[0],'AP','Test average precision'),(axes[1],'F1','Test F1 · fixed validation threshold')]:
        ax.barh(y-.17,[r['old_'+key] for r in comparisons],height=.3,color='#b8c6d1',label='Previous 5-epoch run')
        ax.barh(y+.17,[r['new_'+key] for r in comparisons],height=.3,color='#2a8b8a',label='Validation-selected repair')
        for i,r in enumerate(comparisons):ax.text(max(r['old_'+key],r['new_'+key])+.015,i,f"{r['old_'+key]:.3f} → {r['new_'+key]:.3f}",va='center',fontsize=8)
        ax.set_xlim(0,1.25);ax.set_xticks([0,.25,.5,.75,1]);ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    axes[0].set_yticks(y,[NAMES[n] for n in names]);axes[0].invert_yaxis();axes[0].legend(loc='lower right',frameon=False,fontsize=8)
    fig.suptitle('Kronos SupervisedAD · predeclared repair rounds',x=.035,ha='left',fontsize=17,fontweight='bold')
    fig.text(.035,.015,'Same historical holdouts · per-dataset choices use validation only · unequal training budgets are disclosed · no cross-dataset average',fontsize=9,color='#586470')
    fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(out/'before_after.png',dpi=180);plt.close(fig)
    pr_html=''
    curve_names=[n for n in names if n in new_curves]
    if curve_names:
        _,old_curves=read_results(args.reference_public,args.reference_private) if args.reference_public or args.reference_private else ({},{})
        fig,axes=plt.subplots((len(curve_names)+2)//3,3,figsize=(13,3.3*((len(curve_names)+2)//3)),squeeze=False)
        for ax,name in zip(axes.flat,curve_names):
            if name in old_curves:
                c=old_curves[name];ax.step(c['recall'],c['precision'],where='post',color='#9fadb9',lw=1.4,label='Previous')
            c=new_curves[name];m=new[name]['test']
            ax.step(c['recall'],c['precision'],where='post',color='#2a8b8a',lw=1.5,label='Selected repair')
            ax.axhline(m['prevalence'],ls='--',color='#ca8832',lw=.8)
            ax.scatter([m['recall']],[m['precision']],color='#bd5158',s=20,zorder=3)
            ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Recall',ylabel='Precision',title=NAMES[name]);ax.grid(alpha=.12)
            ax.legend(fontsize=7,frameon=False,loc='upper right')
        for ax in list(axes.flat)[len(curve_names):]:ax.set_visible(False)
        fig.suptitle('Precision–recall comparison · red dot = fixed validation threshold',fontsize=15,fontweight='bold')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'precision_recall_comparison.png',dpi=160);plt.close(fig)
        code=base64.b64encode((out/'precision_recall_comparison.png').read_bytes()).decode()
        pr_html=f'<section><h2>精确率–召回率曲线</h2><img src="data:image/png;base64,{code}" alt="修复前后精确率召回率曲线"><p>曲线由完整测试分数计算；绘图保留召回率区间边界及固定阈值点。橙虚线为异常占比，红点为选中方案的固定验证阈值。</p></section>'
    headings=['数据集','检测单位','选中轮次','测试/异常数','旧 AP','新 AP','AP 变化','旧 F1','新 F1','旧召回','新召回','旧误报','新误报']
    rows=[]
    for r in comparisons:
        rows.append([NAMES[r['dataset']],{'time point':'时间点','transaction':'交易','transaction node':'交易节点','whole sequence':'整段序列'}[r['unit']],r['selected_round'],f"{r['n']:,} / {r['positives']:,}",f"{r['old_AP']:.4f}",f"{r['new_AP']:.4f}",f"{r['delta_AP']:+.4f}",f"{r['old_F1']:.4f}",f"{r['new_F1']:.4f}",f"{100*r['old_recall']:.2f}%",f"{100*r['new_recall']:.2f}%",f"{r['old_FP']:,}",f"{r['new_FP']:,}"])
    table='<thead><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody>'
    detail=''
    for name,selection in trials.items():
        entries=''.join(f"<tr><td>{t['config']['name']}</td><td>{t['config']['epochs']}</td><td>{t['validation']['average_precision']:.5f}</td><td>{t['validation']['f1']:.5f}</td></tr>" for t in selection['trials'])
        detail+=f'<details><summary>{NAMES[name]} · 所有验证轮次</summary><table><tr><th>方案</th><th>最大轮数</th><th>验证 AP</th><th>验证 F1</th></tr>{entries}</table></details>'
    confirmation_html=''
    for name,data in confirmations.items():
        confirmation_html+=f'<details><summary>{NAMES[name]} · 固定方案的额外种子确认</summary><table><tr><th>种子</th><th>方案</th><th>测试 AP</th><th>测试 F1</th></tr>'
        for r in data['runs']:confirmation_html+=f"<tr><td>{r['seed']}</td><td>{r['config']['name']}</td><td>{r['test']['average_precision']:.5f}</td><td>{r['test']['f1']:.5f}</td></tr>"
        confirmation_html+='</table></details>'
    reference_html=''
    if local_reference and 'ieee_cis' in new:
        a=local_reference['metrics'];b=new['ieee_cis']['test']
        reference_html=f'<section><h2>IEEE-CIS 的同设备复核</h2><p>旧结构与修复版均在本机 MPS 上训练／评估：AP {a["average_precision"]:.4f} → {b["average_precision"]:.4f}，F1 {a["f1"]:.4f} → {b["f1"]:.4f}。上方主表使用此前公布的 CPU 历史结果作为修复前记录，此处另列同设备参照。</p></section>'
    encoded=base64.b64encode((out/'before_after.png').read_bytes()).decode()
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kronos 多轮改进结果</title><style>body{{font:15px/1.7 system-ui;background:#f4f7fa;color:#203446;margin:0}}main{{max-width:1400px;margin:auto;padding:32px}}section,details{{background:white;border-radius:12px;padding:24px;margin:18px 0}}img{{width:100%}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e0e7ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}input{{padding:10px;border:1px solid #b4c2cc;border-radius:6px}}</style><main><h1>Kronos SupervisedAD · 多轮修复实测</h1><p>已完成 {len(names)} 项。同一历史测试集上的前后对照，保留退步结果。</p><img src="data:image/png;base64,{encoded}" alt="修复前后 AP 与 F1"><section><input id="filter" placeholder="筛选数据集"><div class="scroll"><table id="comparison">{table}</table></div></section><section><p>R0 为旧方法 5 轮；R1 为同预算的原始特征残差通路；R2 为更快学习率、最多 12 轮；R3 为最多 12 轮、正类权重上限 100。R2/R3 同时改变多个训练设置，不能把它们的全部提升归因于结构变化。各数据先按验证 AP、验证 F1 及较早方案的顺序选定，再评估测试。R0 仍可被选中。</p><p>不同检测单位不混算总平均。大交易数据维持相同的标签无关训练／验证抽样上限，完整测试段不变。IEEE-CIS 原始和派生记录留在本机；报告不导出其类别词表。加权 BCE 分数未经校准。</p><p>旧测试结果此前已公布，本次是历史留出集复测，不是全新外部盲测。额外种子只确认已固定方案，不用于事后更换方案。改进需要同时看 AP、F1、召回及误报，不能只挑其中一项。</p></section>{reference_html}{pr_html}{detail}{confirmation_html}</main><script>document.getElementById('filter').addEventListener('input',function(){{document.querySelectorAll('#comparison tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()));}})</script></html>'''
    (out/'report.html').write_text(document)
    print(json.dumps(dict(datasets=len(comparisons),AP_improved=sum(r['delta_AP']>1e-8 for r in comparisons),F1_improved=sum(r['delta_F1']>1e-8 for r in comparisons),output=str(out))))


if __name__=='__main__':main()
