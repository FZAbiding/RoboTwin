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
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.training_record import read_training_status


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
    supervisor=read(r/'supervisor/status.json');training_rows=[]
    for seed in [0,1]:
        path=r/'training'/f'seed-{seed}'/'status.json'
        v=read_training_status(path) if path.exists() else {};epochs=v.get('epochs',[])
        loss=epochs[-1].get('validation',{}).get('loss') if epochs else None
        display_status=v.get('status','待运行')
        for job in supervisor.get('jobs',[]):
            if job.get('name')==f'train-seed-{seed}' and job.get('exit_code') not in (None,0):
                display_status=f'异常退出{job["exit_code"]}；原状态可能未刷新'
        training_rows.append(f'<tr><td>{seed}</td><td>{esc(display_status)}</td><td>{len(epochs)}/6000轮；{v.get("updates",0)}/18000更新</td><td>{esc(round(loss,6) if loss is not None else "—")}</td><td>{link(r/"training"/f"seed-{seed}"/"status.json","训练证据")} · {link(r/"configs"/f"train-seed-{seed}.json","配置")}</td></tr>')
    checks=read(r/'checks/prerequisites.json')
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
    followup=read(r/'checks/worker-failure-followup.json')
    prior=read(r/'snapshots/prior-study.json')
    if followup:
        diagnostic=Path(followup['diagnostic_run'])
        stress=read(diagnostic/'native-reader-2/status.json')
        evaluation+=f'<p>长训练首尝试因worker SIGSEGV在445个完整epoch、1336次更新处失败；监督器与模型比较按设计停止，未生成新策略成绩。新诊断记录worker原生故障栈及最后样本，注入故障和随机样本不变两项回归通过；原50+派生50个HDF5哈希不变。GPU外原生读取压力验证：{esc(stress.get("status","待运行"))}，{stress.get("epochs",0)}轮/{stress.get("samples",0)}样本。{link(diagnostic/"checks/data-hash-audit.json","数据哈希")} · {link(diagnostic/"logs/diagnostics-regression.log","故障记录回归")} · {link(diagnostic/"native-reader-2/status.json","原生读取压力状态")}</p>'
        evaluation+='<p>诊断首个压力进程还暴露了独立的AF_UNIX路径过长错误；保留失败日志，改用/home下较短专用TMPDIR，并在长作业前实际建立本地socket验证。此问题不能直接解释先前SIGSEGV，根因仍待现场栈证据。</p>'
    if prior:
        if (r/'snapshots/schedule-amendment.json').exists():
            evaluation+=f'<p>本阶段复用种子0已完整完成的6000轮连续轨迹，先评测它的三个预算点；种子1在新目录从头补齐，再按原两种子均值选择预算并独立确认。旧种子1部分权重保留但不混入新轨迹。模型、数据、预算和选择规则保持；仅调整调度与诊断记录形式。{link(r/"snapshots/schedule-amendment.json","结果出现前的调度调整")}</p>'
        else:
            evaluation+=f'<p>当前为全新重试目录；复用已冻结的比较/确认清单，没有重新抽选种子。两训练种子从头初始化，没有把权重加载称为优化器精确续训；模型、数据、训练更新预算及选择规则保持。仅新增worker故障观察与IPC检查。{link(Path(prior["failed_training"]),"原训练失败记录")} · {link(r/"snapshots/prior-study.json","重试差异记录")}</p>'
        evaluation+=f'<p>新增诊断的真实短训练与关闭诊断的对照都完成6次更新，初始验证loss相同，更新后权重最大差约4.72e−5，不能声称位级一致。{link(r/"checks/diagnostic-smoke-audit.json","短训练验收")} · {link(r/"checks/smoke-numerical-comparison.json","观察器数值对照")} · {link(r/"checks/worker-resources.jsonl","worker CPU/内存/FD现场")}。资源采样只读取本任务父子进程，不修改CPU亲和性或全局配置。</p>'
    manifests_audit=read(r/'checks/manifests-audit.json')
    checkpoint_audit=read(r/'checks/checkpoint-s0-u3000-audit.json')
    completed_seed0=read(r/'checks/training-seed-0-audit.json')
    if manifests_audit or checkpoint_audit:
        evidence_rows=[]
        if manifests_audit:
            evidence_rows.append(f'<tr><td>完整专家清单</td><td>200个种子与训练集及另一队列互不重叠；2400个轨迹数组均有限，评测配置和冻结证明一致</td><td>{link(r/"checks/manifests-audit.json","全清单验收")}</td></tr>')
        if checkpoint_audit:
            evidence_rows.append(f'<tr><td>种子0 · 3000更新</td><td>CPU重载344个有限张量、14维接口与原归一化一致；当前完整轨迹状态见训练表</td><td>{link(r/"checks/checkpoint-s0-u3000-audit.json","权重与统计验收")}</td></tr>')
        if completed_seed0:
            evidence_rows.append(f'<tr><td>种子0 · 完整6000轮</td><td>18000次更新完成；三个预算点权重有限，原始与派生各50文件哈希不变</td><td>{link(r/"checks/training-seed-0-audit.json","完整轨迹单独验收")}</td></tr>')
        evaluation+='<figure><div class="ve-table"><table><thead><tr><th>提前完成的检查</th><th>实际结论</th><th>证据</th></tr></thead><tbody>'+''.join(evidence_rows)+'</tbody></table></div><figcaption>完整清单和中间权重已独立核查；它们不能替代六千轮轨迹完成及真实策略闭环。</figcaption></figure>'
    if (r/'checks/evaluation-contract-review.json').exists():
        evaluation+=f'<p>输入时序复核：save_freq=15是专家路径采样步距，frequency=30是当前ACT不读取的元数据，真实动作由250Hz原生TOPP路径推进。固定队列逐种子调用会重置缓存计数，现按全局回合序号恢复原清理节奏并保留异常强制清理；未改动作、模型输入或任务判据。{link(r/"checks/evaluation-contract-review.json","时序与缓存复核")}。缓存标志回归不是物理策略成绩，真实长队列释放仍由后续逐回合审计确认。</p>'
    if (r/'checks/official-ws-port-lifecycle.json').exists():
        evaluation+=f'<p>连续切换权重的服务预检复核：官方WebSocket传输关闭后，普通bind实测返回EADDRINUSE；与官方asyncio服务一致的SO_REUSEADDR可立即绑定，同时回归确认活动监听端口仍被拒绝。这里只修复本任务预检，不设置SO_REUSEPORT、不停止其他服务；此传输检查没有加载模型，不能作为推理成绩。{link(r/"checks/official-ws-port-lifecycle.json","官方传输实测")} · {link(r/"logs/policy-port-regression.log","端口生命周期回归")}</p>'
    outcome=read(r/'checks/prior-attempt-outcome.json')
    if outcome:
        evaluation+=f'<figure><div class="ve-table"><table><thead><tr><th>原尝试</th><th>退出与进度</th><th>证据及处理</th></tr></thead><tbody><tr><td>种子0</td><td>退出0；6000轮/18000更新完成</td><td>复用同一条连续轨迹；三个权重在CPU验收，随后真实评测</td></tr><tr><td>种子1</td><td>退出{outcome["seed1_exit"]}；追加日志{outcome["seed1_logged_epochs"]}轮/{outcome["seed1_last_logged_updates"]}更新，状态文件停在{outcome["seed1_status_file_epochs"]}轮</td><td>主进程JSON写入时段错误；worker故障文件为空，根因未证明；原件与部分权重保留</td></tr></tbody></table></div><figcaption>异常退出时状态文件可能未刷新，以退出码与追加历史交叉核对，不能把原文件的running当作仍在训练。{link(r/"checks/prior-attempt-outcome.json","交叉核对")}</figcaption></figure>'
        evaluation+=f'<p>新记录格式保留完整epochs.jsonl，每轮只原子替换简短状态，旧内嵌格式继续可读；这是减少累积日志重复写入，不能宣称修复段错误。原出错时的持久化JSON重构后128次纯Python序列化未复现，不能排除原进程原生堆损坏或硬件因素。{link(r/"checks/json-replay-audit.json","JSON重放边界")}</p>'
    waiting=pipeline.get('waiting_for_resources')
    if waiting:
        evaluation+=f'<p>资源排队：{esc(waiting["job"])} 当前空闲显存{waiting["gpu_free_mib"]}MiB，需要{waiting["minimum_gpu_free_mib"]}MiB；磁盘空闲{waiting["disk_free_bytes"]/1024**3:.2f}GiB，需要{waiting["minimum_disk_free_bytes"]/1024**3:.0f}GiB。满足条件后自动启动，等待也计入本任务保守48小时窗口。</p>'
    recovery=read(r/'checks/cache-space-recovery.json')
    if recovery:
        evaluation+=f'<p>本阶段磁盘一度不足：已先保存逐文件名称/大小/SHA，再仅释放本任务安装时的旧Conda压缩包及pip HTTP下载缓存，共{recovery["files_deleted"]}文件、{recovery["download_bytes_unlinked"]/1024**3:.2f}GiB。解压依赖、安装环境、编译扩展、原始资产/数据、权重和失败记录保留，清理后关键环境导入验证通过。{link(r/"checks/cache-space-recovery.json","回收结果")} · {link(r/"snapshots/evicted-download-cache-manifest.json","下载缓存原哈希清单")} · {link(r/"checks/environment-after-cache-recovery.json","安装环境核对")}</p>'
    conflict=read(r/'checks/gpu-resource-conflict.json')
    if conflict and not conflict.get('service_stopped'):
        evaluation+=f'<p>当前大显存服务为另一个工作目录的serve_policy.py，PID {conflict["pid"]}。已准备本任务队列并请求明确授权后再考虑SIGTERM；未自行终止其他作业。{link(r/"checks/gpu-resource-conflict.json","进程身份与授权状态")}</p>'
    if comparison.get('models'):
        rows=[];bars=[]
        for i,v in enumerate(comparison['models']):
            label=f"种子{v['training_seed']} · {v['updates']}更新"
            rows.append(f'<tr><td>{esc(label)}</td><td>{v["successes"]}/100</td><td>{link(r/"checks"/(v["name"]+"-audit.json"),"逐种子/视频验收")}</td></tr>')
            y=25+i*38
            bars.append(f'<text x="0" y="{y+12}" fill="currentColor" font-size="12">{esc(label)}</text><rect x="165" y="{y}" width="300" height="18" fill="currentColor" opacity="0.12"/><rect x="165" y="{y}" width="{v["successes"]*3}" height="18" fill="currentColor"/><text x="475" y="{y+12}" fill="currentColor" font-size="12">{v["successes"]}/100</text>')
        height=50+38*len(bars)
        evaluation+=f'<figure><svg viewBox="0 0 550 {height}" role="img" aria-label="冻结比较队列的各检查点真实成功数">'+''.join(bars)+'</svg><figcaption>同一100个专家验证场景逐模型比较；条形长度表示真实策略成功数，不能解释为无条件任务成功率。</figcaption></figure><figure><div class="ve-table"><table><thead><tr><th>检查点</th><th>比较集成功</th><th>证据</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><figcaption>保留全部已执行模型及失败场景，按预登记规则选择预算。</figcaption></figure>'
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
<p>当前/home可用{resources['home_disk_free_bytes']/1024**3:.2f}GiB；{link(r/'checks/resource-samples.jsonl','每分钟资源采样')}。GPU时长按本任务墙钟占用保守计费，预算48小时；训练新作业保留4GiB、评测3GiB空间，运行中低于1.5GiB停止并保留日志。</p>
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
