#!/usr/bin/env python3
"""Append a live experiment section to the existing local research report."""
from pathlib import Path
import argparse,json,datetime,html,os,re,subprocess,shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--report-source',type=Path,required=True);p.add_argument('--renderer',type=Path,required=True);p.add_argument('--report-root',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve();src=a.report_source.resolve();report=src.parent
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    def esc(value):return html.escape(str(value))
    def link(path,label):return f'<a href="{esc(os.path.relpath(path,report))}">{esc(label)}</a>' if path.exists() else '待产物'
    backup=r/'snapshots/report-before-controls.src.html'
    if not backup.exists():shutil.copy2(src,backup)
    supervisor=read(r/'supervisors/controls/status.json');phase=read(r/'status.json')
    phase.update(status=supervisor.get('status','preparing'),active_jobs=[{'name':v['name'],'pid':v['pid']} for v in supervisor.get('jobs',[]) if 'exit_code' not in v],server_pid=supervisor.get('server_pid') if supervisor.get('status')=='running' else None,updated=datetime.datetime.now().isoformat(timespec='seconds'))
    (r/'status.json').write_text(json.dumps(phase,indent=2))
    comparison=read(r/'checks/control-comparison.json');rows=[];nodes=[];names=['baseline-1','reset-1','baseline-2','reset-2']
    for i,name in enumerate(names):
        v=read(r/'eval'/name/'status.json');audit=read(r/'checks'/f'{name}-audit.json');reset=read(r/'configs'/f'{name}.json').get('diagnostic_options',{}).get('reset_planners');status='已交叉验收' if audit.get('passed') else v.get('status','待运行');trials=v.get('trials',[]);success=sum(bool(t['success']) for t in trials)
        rows.append(f'<tr><td>{esc(name)}</td><td>{esc(status)}</td><td>{"重置四个MotionGen" if reset else "保留原生复用"}</td><td>{len(v.get("candidates",[]))}/49候选；{len(trials)}真实策略回合；{success}成功</td><td>{link(r/"checks"/f"{name}-audit.json","验收")} · {link(r/"configs"/f"{name}.json","有效配置")}</td></tr>')
        nodes.append(f'<ve-node id="pc-{name}" at="{i+1} 1" sub="{esc(status)}">{esc(name)}</ve-node>')
    edges=''.join(f'<ve-edge from="pc-{names[i]}" to="pc-{names[i+1]}" label="顺序运行"></ve-edge>' for i in range(3));details=''
    if comparison:
        details=f'<p>重复两轮的专家接纳集合变化数：原生{comparison["baseline_replica_eligibility_changes"]}，重置{comparison["reset_replica_eligibility_changes"]}。候选队列相同，模型固定；策略成功数会随专家接纳集合变化，不能把条件成功率变化当作模型学习提升。{link(r/"checks/control-comparison.json","逐种子与轨迹哈希比较")}</p>'
        details+='<p>重置两轮均接纳21个种子、策略13成功；原生两轮分别19个接纳/11成功与20个接纳/12成功。共同接纳种子的策略成功和动作数不变。四轮的23个base1候选均未通过原生专家筛选。重置仅提高本次接纳集合的观察重复性，不能据两次重复宣称确定性或解决base1问题；专家路径数值与形状仍有差异，不能用接纳集合相同替代轨迹重复。</p>'
    query=read(r/'planner-query-r2/summary.json')
    if query.get('status')=='complete':
        count=sum(bool(v['success']) for v in query['queries'])
        details+=f'<p>冻结100001同一起点/目标的批量路径，裁剪有效528步后末端位置误差约2微米、方向误差为0；原始返回每条填充5000步，因此只比较数组长度不能选择真实最短路径。单目标时间细化尺度0.85–1.05、每项重复2次，成功{count}/10，均未解决原失败；没有执行或接纳这些查询路径。{link(r/"planner-query-r2/summary.json","冻结查询与末端检查")}</p>'
        details+=f'<p>首次新增查询记录器误把单目标轨迹按批量切片，触发JointState does not have horizon；修正仅处理批量路径，新目录重跑成功。原失败日志保留。{link(r/"planner-query/summary.json","原失败记录")}</p>'
    section_id='planner-controls-20261106'
    block=f'<section id="{section_id}"><div class="sec-head"><p class="ve-label">本地保存与规划器受控对照 · {datetime.datetime.now().isoformat(timespec="seconds")}</p><h3>固定模型与候选序列，分离验证规划器重置</h3></div><figure><ve-flow label="原生和重置条件交错运行，各49个相同候选，均由真实专家筛选和真实ACT策略执行。">'+''.join(nodes)+edges+'</ve-flow><figcaption>只改变规划器状态重置；各轮独立评测进程，首个新建场景保持原生初始化，后续复用场景按条件重置单目标与批量MotionGen。</figcaption></figure><figure><div class="ve-table"><table><thead><tr><th>轮次</th><th>实际状态</th><th>条件</th><th>实时结果</th><th>证据</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><figcaption>候选预算审计不是完成了原生请求回合数；49个候选全部处理后明确标记预算结束，未放宽筛选或任务判据。</figcaption></figure><p>用户已授权本地提交和最长48小时GPU资源，不推送远端、不创建PR。保存点：RoboTwin 886b034、XPolicyLab 6c1bb42f；新有界实验工具6736ba6。实验目录 '+esc(str(r))+'。</p><p>真实3候选试跑已通过：2个策略回合、490次动作，视频及原生成功分量交叉验收通过。每个候选保留专家原始路径副本和SHA256。'+link(r/'checks/smoke-audit.json','试跑验收')+' · '+link(r/'status.json','运行与授权记录')+' · '+link(r.parents[2]/'experiments/README.md','受Git管理的复现问题记录')+'</p>'+details+'</section>'
    text=src.read_text()
    if f'id="{section_id}"' in text:text=re.sub(r'<section id="'+section_id+r'">.*?</section>',lambda _:block,text,flags=re.S)
    else:text=text.replace('<div class="deep" id="next-stage">',block+'\n<div class="deep" id="next-stage">',1)
    assert text.count(f'id="{section_id}"')==1;src.write_text(text)
    result=subprocess.run(['node',str(a.renderer),str(src),'--root',str(a.report_root),'--no-check'],capture_output=True,text=True);(r/'logs/report-update.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    print('control report updated',flush=True)


if __name__=='__main__':main()
