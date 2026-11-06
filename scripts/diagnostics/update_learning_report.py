#!/usr/bin/env python3
"""Update the authorized research report from concrete native study artifacts."""
from pathlib import Path
import argparse
import datetime
import html
import json
import os
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','report-source','renderer','report-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();r=a.run_dir.resolve();src=a.report_source.resolve()
    def read(p):return json.loads(p.read_text()) if p.exists() else {}
    def esc(s):return html.escape(str(s))
    def link(p,label):return f'<a href="{esc(os.path.relpath(p,src.parent))}">{esc(label)}</a>' if p.exists() else '待运行'
    backup=r/'snapshots/report-before-learning.src.html'
    if not backup.exists():shutil.copy2(src,backup)
    manifest_rows=[]
    for name in ['manifest-smoke','comparison','confirmation']:
        v=read(r/'manifests'/name/'status.json')
        coverage='；'.join(f'base{i}={sum(row["model_id"]==i for row in v.get("accepted",[]))}' for i in [0,1])
        manifest_rows.append(f'<tr><td>{esc(name)}</td><td>{esc(v.get("status","待运行"))}</td><td>{len(v.get("candidates",[]))}</td><td>{len(v.get("accepted",[]))}；{coverage}</td><td>{link(r/"manifests"/name/"manifest.json","冻结证明")} · {link(r/"configs"/f"{name}.json","配置")}</td></tr>')
    training_rows=[]
    for seed in [0,1]:
        v=read(r/'training'/f'seed-{seed}'/'status.json');epochs=v.get('epochs',[])
        loss=epochs[-1].get('validation',{}).get('loss') if epochs else None
        training_rows.append(f'<tr><td>{seed}</td><td>{esc(v.get("status","待运行"))}</td><td>{len(epochs)}/6000轮；{v.get("updates",0)}/18000更新</td><td>{esc(round(loss,6) if loss is not None else "—")}</td><td>{link(r/"training"/f"seed-{seed}"/"status.json","训练证据")} · {link(r/"configs"/f"train-seed-{seed}.json","配置")}</td></tr>')
    supervisor=read(r/'supervisor/status.json');checks=read(r/'checks/prerequisites.json')
    pipeline=read(r/'evaluation-pipeline/status.json')
    resources={'time':datetime.datetime.now().isoformat(timespec='seconds'),
               'home_disk_free_bytes':shutil.disk_usage(r).free}
    try:
        gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used,memory.free,power.draw,temperature.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        resources['gpu_util_used_free_power_temp_csv']=gpu.stdout.strip()
        resources['nvidia_smi_exit_code']=gpu.returncode
    except subprocess.TimeoutExpired:resources['nvidia_smi_exit_code']='timeout'
    with (r/'checks/resource-samples.jsonl').open('a') as file:file.write(json.dumps(resources)+'\n')
    phase=read(r/'status.json')
    phase.update(status=pipeline.get('status',supervisor.get('status','preparing')),
                 training_supervisor_pid=supervisor.get('pid'),evaluation_pipeline_pid=pipeline.get('pid'),
                 active_training_jobs=[{'name':v['name'],'pid':v['pid']} for v in supervisor.get('jobs',[]) if 'exit_code' not in v],
                 updated=datetime.datetime.now().isoformat(timespec='seconds'))
    (r/'status.json').write_text(json.dumps(phase,indent=2))
    comparison=read(r/'checks/comparison-results.json');confirmation=read(r/'checks/confirmation-results.json')
    evaluation=''
    if comparison.get('models'):
        rows=[];bars=[]
        for i,v in enumerate(comparison['models']):
            label=f"种子{v['training_seed']} · {v['updates']}更新"
            rows.append(f'<tr><td>{esc(label)}</td><td>{v["successes"]}/100</td><td>{link(r/"checks"/(v["name"]+"-audit.json"),"逐种子/视频验收")}</td></tr>')
            y=25+i*38
            bars.append(f'<text x="0" y="{y+12}" fill="currentColor" font-size="12">{esc(label)}</text><rect x="165" y="{y}" width="300" height="18" fill="currentColor" opacity="0.12"/><rect x="165" y="{y}" width="{v["successes"]*3}" height="18" fill="currentColor"/><text x="475" y="{y+12}" fill="currentColor" font-size="12">{v["successes"]}/100</text>')
        height=50+38*len(bars)
        evaluation=f'<figure><svg viewBox="0 0 550 {height}" role="img" aria-label="冻结比较队列的各检查点真实成功数">'+''.join(bars)+'</svg><figcaption>同一100个专家验证场景逐模型比较；条形长度表示真实策略成功数，不能解释为无条件任务成功率。</figcaption></figure><figure><div class="ve-table"><table><thead><tr><th>检查点</th><th>比较集成功</th><th>证据</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><figcaption>保留全部已执行模型及失败场景，按预登记规则选择预算。</figcaption></figure>'
    if confirmation:
        evaluation+=f'<p>独立确认：所选模型{confirmation["selected"]["successes"]}/100，同训练种子3k更新基线{confirmation["same_seed_3000_update_baseline"]["successes"]}/100；配对新增成功{confirmation["paired_wins"]}，新增失败{confirmation["paired_losses"]}，差值{confirmation["success_difference_percentage_points"]}个百分点，精确双侧配对检验p={confirmation["paired_exact_two_sided_p"]:.6g}。若检查点相同，未发生预算提升比较。{link(r/"checks/confirmation-results.json","确认逐种子结果")}</p>'
    section='learning-study-20261106'
    block=f'''<section id="{section}"><div class="sec-head"><p class="ve-label">固定数据与冻结队列 · {datetime.datetime.now().isoformat(timespec='seconds')}</p><h3>用同一批真实场景比较训练更新预算</h3></div>
<figure><ve-flow label="先冻结原生专家通过场景，再对两条连续训练轨迹的检查点做配对比较。"><ve-node id="ls-data" at="1 1" sub="原50条验证数据">数据固定</ve-node><ve-node id="ls-train" at="2 1" sub="种子0、1；3k/9k/18k更新">连续原生训练</ve-node><ve-node id="ls-compare" at="3 1" sub="300000起；100个专家成功种子">固定队列比较</ve-node><ve-node id="ls-confirm" at="4 1" sub="400000起；独立100个种子">最终确认</ve-node><ve-edge from="ls-data" to="ls-train" label="保持输入与动作契约"></ve-edge><ve-edge from="ls-train" to="ls-compare" label="同场景配对"></ve-edge><ve-edge from="ls-compare" to="ls-confirm" label="预登记规则选择"></ve-edge></ve-flow><figcaption>比较集用于选择预算，独立确认集检验所选模型；重复使用的场景不计作新增独立样本。</figcaption></figure>
<figure><div class="ve-table"><table><thead><tr><th>专家清单</th><th>状态</th><th>已尝试候选</th><th>原生验收通过/资产覆盖</th><th>证据</th></tr></thead><tbody>{''.join(manifest_rows)}</tbody></table></div><figcaption>原生采集主循环要求规划成功、真实物理任务成功和规划关节合法，保留初态、专家终态、路径及哈希；这些真值不会传入视觉模型。若base1无入选，模型成绩只能代表当前专家条件下的base0覆盖。</figcaption></figure>
<figure><div class="ve-table"><table><thead><tr><th>训练种子</th><th>状态</th><th>实际预算</th><th>最近验证loss</th><th>证据</th></tr></thead><tbody>{''.join(training_rows)}</tbody></table></div><figcaption>两种子沿各自一条连续原生优化轨迹保存3k、9k和18k更新权重；loss不能替代真实闭环成功。</figcaption></figure>
<p>保持640×480三相机、14维动作、batch16、学习率1e−5、chunk50、KL10与原任务判据。数据仍为原生40/10划分；原生归一化使用全部50条和末帧填充，包含验证集，按既定协议保留并披露。</p>
<p>本组显式使用spawn及持久数据worker，并重新训练自己的1000轮参考点；历史权重作为外部参考，不能声称位级复现。先验收真实短训练的更新数、有限权重与worker复用，再开启长训练。监督器提供超时、无进展栈诊断、磁盘保护和本任务进程组管理。</p>
<p>当前作业监督状态：{esc(supervisor.get('status','准备中'))}；评测流水线：{esc(pipeline.get('status','准备中'))}。{link(r/'supervisor/status.json','PID与耗时台账')} · {link(r/'evaluation-pipeline/status.json','评测排队/结果状态')} · {link(r/'checks/prerequisites.json','运行前验收')} · {link(r/'snapshots/preregistered-plan.json','预登记选择规则')}。训练和完整确认尚未完成时，不发布策略提升结论。</p>
<p>当前/home可用{resources['home_disk_free_bytes']/1024**3:.2f}GiB；{link(r/'checks/resource-samples.jsonl','每分钟资源采样')}。GPU时长按本任务墙钟占用保守计费，预算48小时；新作业至少保留4GiB空间，运行中低于1.5GiB停止并保留日志。</p>
{evaluation}
<p>复现问题与处理：原生专家重复筛选会改变接纳队列，因此在观察新模型结果前冻结两组专家证明；采集seed.txt采用已回归验证的原子替换。历史原生卡住的根因仍未确定，spawn协议是本组显式实验条件，不能据此宣称已定位根因。HTML生成成功不代表浏览器版面验收，既有SVGPoint工具故障仍单独记录。</p>
<p>实验目录：{esc(r)}。原始数据、历史结果和失败尝试全部保留；只做本地Git保存，不推送或创建PR。</p></section>'''
    text=src.read_text()
    if f'id="{section}"' in text:text=re.sub(r'<section id="'+section+r'">.*?</section>',lambda _:block,text,flags=re.S)
    else:text=text.replace('<div class="deep" id="next-stage">',block+'\n<div class="deep" id="next-stage">',1)
    active='、'.join(v['name'] for v in supervisor.get('jobs',[]) if 'exit_code' not in v) or '无采集/训练作业'
    index=f'<p id="live-study-index">最新实测：<a href="#execution-progress-20261105">50条数据/训练/真实闭环</a> · <a href="#planner-controls-20261106">规划器四组对照已验收</a> · <a href="#learning-study-20261106">冻结队列与双种子预算实验</a>。当前：{esc(active)}；评测流水线{esc(pipeline.get("status","准备中"))}。各阶段的失败、修正、有效配置和证据保留在对应章节。</p>'
    if 'id="live-study-index"' in text:
        text=re.sub(r'<p id="live-study-index">.*?</p>',lambda _:index,text,flags=re.S)
    else:
        marker='<section id="stack"><div class="sec-head"><p class="ve-label">环境</p><h2>主分支在 SAPIEN 中运行，Isaac 可用于制作规划配置</h2></div>'
        assert marker in text;text=text.replace(marker,marker+index,1)
    text=text.replace('<td>XPolicyLab @ fa431ecd893e</td>',
                      '<td>上游fa431ecd893e；本地对齐修正6c1bb42f</td>')
    assert text.count(f'id="{section}"')==1;src.write_text(text)
    result=subprocess.run(['node',str(a.renderer),str(src),'--root',str(a.report_root),'--no-check'],capture_output=True,text=True)
    (r/'logs/report-update.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    print('learning study report updated',flush=True)


if __name__=='__main__':main()
