# RoboTwin 本机实验记录

本目录保存受 Git 管理的复现方法、问题与结论索引。原始资产、数据、权重、视频、逐动作日志和有效配置保存在 `data/local-runs/`，不将大文件加入 Git。每轮必须使用全新输出目录，保留失败记录，记录运行版本、配置及产物哈希。

## 当前授权与版本边界

2026-11-06 用户明确授权：本地提交 Git 保存，禁止推送远端和创建 PR；继续规划器一致性验证及后续受控训练实验，可使用最长 48 小时 GPU 时间，需持续汇报。此授权覆盖旧规则中的本地提交限制。用户已要求不再以 HANDOFF.md 作为执行依据。

初始代码：RoboTwin `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`，XPolicyLab `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。所有后续改动和实验版本另行记录；原始成绩不重标。

## 已完成链路

任务 `lift_pot`，物理机器人 `aloha-agilex`，独立配置完整沿用 `demo_clean` 初态分布和原始成功判据。`arx_x5` 仅提供双臂 `6+1, 6+1` 数据维度映射，不代表验证了 X5 机器人。

- 环境：Mamba `robotwin`，Python 3.10.22，SAPIEN 3.0.0b1，Torch 2.4.1+cu121，NumPy 1.26.4，MPLib 0.2.1，CuRobo v0.7.8。环境内 CUDA 12.1 / GCC12；系统驱动和 CUDA 链接未改。
- 专家筛选和原生回放：50 条成功示范，5,553 对 state/action。全部为 kitchenpot/base0；原件包含四相机，ACT 使用 head/right_wrist/left_wrist 三路。
- 转换：共享 RGB 解码器，640×480，无损 LZF、按帧分块，约 4.29 GB；16,659 张派生图像逐帧比对通过，原始与派生文件均有 SHA256。
- ACT：40/10 回合划分，batch16、chunk50、lr1e-5、KL10、hidden512、FF3200、temporal aggregation；正式训练 1,000 epoch、3,000 次更新。沿用上游全部50回合、末帧补到127的归一化统计；验证集参与了归一化。
- 权重新进程重载、reset、WebSocket 与直接推理动作一致；真实闭环20回合12成功、8失败，4,259次动作。成绩条件于原生专家筛选：49候选、20入选。并非49个候选的无条件成功率。

正式权重 SHA256：`08429b6e19f482cb166a468c932e81a423db00544d4b7341c76b403b740ffa3c`。

## 产物与报告

- 环境/最小回合：`data/local-runs/stage12-20261104-222636/`
- 50条专家回放：`data/local-runs/stage3-20261105-001152/`
- 正式训练和全链路验收：`data/local-runs/fullchain-20261105-005503/checks/fullchain-audit.json`
- 权重：`data/local-runs/fullchain-20261105-005503/checkpoints/formal-r2/`
- 20回合逐动作与视频索引：`data/local-runs/fullchain-20261105-005503/eval/formal-eval-r3/`
- 稳定性与失败诊断：`data/local-runs/diagnosis-20261106-181434/`
- 总报告：`/home/xense/src/xensim-bench/outputs/research/robotwin-overview-20261104/robotwin.html`，章节 `#execution-progress-20261105`、`#diagnosis-20261106`。源码为相邻 `robotwin.src.html`。

这些绝对路径是本机索引。迁移机器时需另行准备原始数据和权重，并复制有效配置、改为全新输出目录；Git 本身不包含大文件。

## 遇到的问题与处理

1. **CuRobo 扩展编译失败。** 固定同一源码与依赖，以 `NVCC_APPEND_FLAGS=--threads=1` 重新构建通过；未升级系统 CUDA 或替换上游源码。
2. **采集断点清单被中断覆盖。** 从成功日志、轨迹序号和哈希恢复37条已保存规划路径，保留中断原件；随后完成原生回放验收。监控器误判连续失败的问题单独修正，未放宽任务成功判据。
3. **ACT 多偏移一帧。** 原始 RoboTwin action[t] 已是 state[t] 对应的下一采样目标，旧读取器又取 action[t-1]。仅为已知 RoboTwin 派生数据写入 `action_alignment`，读取器使用现有对齐分支。真实 k=42 回归及未标记旧数据兼容测试通过。
4. **磁盘不足风险。** 保留原始 HDF5 和640×480输入，以可选无损LZF压缩派生图像；压缩前后像素严格一致。临时目录和缓存使用任务目录，未清理其他工程。
5. **官方服务 shell 重复激活 conda 失败。** 编译器激活脚本在严格 shell 模式下遇到未定义变量；使用同一环境的官方 `XPolicyLab/setup_policy_server.py` 入口，按共享 helper 设置 `ACT_ACTION_DIM`，未修改模型服务实现。
6. **训练/评测原生运行异常。** 首轮正式训练停在714 epoch，监督器超时终止，保留500 epoch权重；在新目录重训1000 epoch完成。一次评测在第二回合第289步 SIGSEGV，另有绘图进程异常；内核同期记录其他Python进程段错误，原因未确认。采用 faulthandler 和数值库单线程后完成20回合；文档绘图与GPU作业隔离。不能把异常终止当正常完成。
7. **TOPP 观察器重复包装。** 原始58条错误日志含跨回合重复包装；逐动作、逐手臂证明精确重复倍数，派生审计得到10次真实回退。原始日志未改；记录器改为幂等包装，本次复测验证每动作每臂只计一次。
8. **同权重复测不是完整20种子的确定性重现。** 复测仍12/20，但专家筛选换掉100037、加入100049。共同19个种子的动作数、成功判定、逐动作数值及实际关节位置均相同，最大数值差为0。两轮不得合并为40个独立样本。
9. **策略失败与资产/规划失败分开。** 同样的8个策略失败中7个全程未同时进入双TCP三厘米范围，1个未同时保持全部条件。base1的23候选中22个规划失败、1个规划完成但物理抓取失败。未改资产、阈值或历史标签。
10. **批量/单目标规划不一致。** base1/100001同一冻结起点和目标，批量规划成功，单目标连续重复失败；100037在独立进程中原专家任务成功。原生状态包括 FINETUNE_TRAJOPT_FAIL、INVALID_PARTIAL_POSE_COST_METRIC 和 IK_FAIL；这些定位失败阶段，不等于已证明几何根因。
11. **网页自动版面检查工具故障。** HTML生成及43幅图图注、110个本机链接检查通过；浏览器检查因工具 SVGPoint 类型错误未完成。旧截图不作为新版本的版面证据。

## 复现入口

先确认实际Python路径、GPU占用和空间，使用原环境及缓存；以下配置是已有实验的记录，不应直接覆盖重跑。

```bash
mamba run -n robotwin python scripts/train_act_recorded.py --config /path/to/new-training.json
mamba run -n robotwin python scripts/eval_act_recorded.py --config /path/to/new-evaluation.json
mamba run -n robotwin python scripts/diagnostics/expert_query_probe.py \
  --run-dir /path/to/new-run --config /path/to/evaluation.json --seed 100001
```

训练与评测记录器观察上游主循环，不复制模型训练或仿真评测循环。评测可选 `diagnostics_module` 指向 `scripts/diagnostics/lift_pot_observer.py`：记录独立的成功条件、实际接触和原生规划返回；这些真值不会传给ACT。

下一阶段先做规划器状态重置和候选执行一致性的分离对照；只有在原成功判据和真实物理验收下验证有效的改动才采纳。随后以固定数据、固定新评测队列做单因素训练对照。全程保留对照组、失败及有效命令。

## 2026-11-06 规划器对照协议

先运行3候选真实试跑，再以相同模型权重交错运行 baseline-1、reset-1、baseline-2、reset-2。每轮检查100000–100048共49个原生候选；所有原生专家接纳的场景都由真实ACT执行，保留400步上限与原始成功判据。重置变体仅在复用场景初始化后调用左右两侧单目标和批量 MotionGen 的官方 reset API，首次新建场景不额外重置。任务、资产和普通基准入口未修改。

`candidate_audit=true` 是单独的预算审计模式：原生请求回合数必须大于候选预算，记录器在下一候选初始化前以专用信号结束原生循环，标记 `candidate_audit_complete`。这不是完成了原生请求回合数，不能作为普通基准成绩发布。预算之外的候选不被初始化，已完成回合与失败完整保存。

`run_policy_experiments.py` 串行管理官方模型服务与真实评测子进程，记录PID、权重/源码哈希、超时、磁盘保护及GPU占用时长。`planner_audit_observer.py` 按条件重置并保存原生专家路径副本；`audit_policy_record.py` 验证候选序列、真实动作、成功条件、视频和路径哈希。

候选资格稳定性、专家物理成功、轨迹数值重复性和策略成功分别报告。后续若采纳规划器代码修复，须另做原生入口回归和新数据验证；不能用新实验覆盖旧成绩。

后续条件执行方案预先记录于 `experiments/plans/20261106-policy-improvement.json`：原50条数据、两训练种子、3k/9k/18k优化更新的原生训练轨迹检查点；新专家验证比较清单与最终确认清单分别从300000、400000起，筛选资格预先冻结，避免各策略重新筛选换队列。仅在证据支持时扩充数据，并独立比较。

原生采集现在支持可选 `start_seed`（默认0）和 `max_seed_attempts`（默认不限）。成功、关节合法性和回放判据不变。`seed.txt` 使用同目录临时文件、flush/fsync和原子替换；替换失败会保留旧清单和待恢复临时文件。回归测试覆盖原格式和故障注入，防止重现历史断点损坏。

长训练记录器新增可选 `checkpoint_update_steps`，在真实优化器完成更新后原子保存模型权重、更新数和SHA256；默认配置不新增检查点。它不恢复优化器状态，不能把模型权重加载伪装成精确续训。预注册学习曲线将使用每个种子的一条连续训练轨迹。

可选 `data_loader` 配置使用spawn和持久worker，保留原数据集、采样器、batch、归一化与原生训练循环；整组新预算采用同一协议，并重新得到其1k epoch参考点。它与历史原生worker协议不是位级复现关系。实际2轮训练、更新数、检查点和worker PID验收通过后才启动长训练；后续原生卡住由SIGUSR1栈诊断和监督器如实记录。

## 2026-11-06 执行结果与持续运行

四组49候选对照均已结束并通过逐动作、视频和原任务判据审计。原生两轮分别接纳19/20、策略成功11/12；重置两轮均接纳21、成功13。共同接纳种子的成功和动作数相同。原生重复的接纳集合差1个种子，重置重复差0个；专家轨迹长度和数值仍有差异，不能宣称全链路确定性。四轮的23个base1候选均未通过原生专家筛选。未把重置变体合并进任务或机器人源码。

冻结100001起点/目标后，原生批量成功路径裁剪至528有效步，末端位置误差约2微米；原始批量输出按5000步填充，不能按填充数组长度选择最短路径。单目标时间细化尺度0.85、0.9、0.95、1.0、1.05各重复2次，0/10成功，均为FINETUNE_TRAJOPT_FAIL。这否定了本次简单时间尺度调整即可修复的假设，但尚未定位全部根因或验证批量路径完整物理可行性。

新增查询记录器首轮把单目标二维轨迹按批量切片，触发“JointState does not have horizon”；仅修正批量提取分支，在`planner-query-r2`新目录重跑完成。原失败日志与退出码保留。一次回归命令误用base Python3.13，因缺Torch导入失败；改为已确认的robotwin Python3.10后两项ACT对齐回归通过，没有向base安装依赖。

新学习实验目录为 `data/local-runs/learning-study-20261106-225739/`。运行前真实验收已通过：2轮训练/6次优化更新、第3次更新权重保存且有限、数据worker PID跨轮复用；两专家种子原生筛选通过；同模型两次冻结队列闭环均2/2成功、每次178动作，初态与动作最大差0，视频完整。它们是工程试跑，不是新模型提升成绩。

`collect_expert_manifest.py` 观察原生collector决定接纳与保存路径，清单记录全部候选、原生规划/物理/关节合法性、场景初态和源码/资产/路径SHA256。`frozen_expert_cohort.py` 在每个指定种子上调用原生`eval_remote_policy(test_num=1)`，保持原生动作和WebSocket循环；按种子隔离视频，重用冻结专家的语言信息，并拒绝指定种子被跳过或替换。专家真值用于资格和诊断，不进入ACT输入。该协议明确区别于每个模型重新运行专家筛选的普通基准。

专家清单采用eval_mode=true，与学习策略场景匹配；lift_pot本身忽略is_test参数，固定配置关闭背景随机化。此等价性限定于当前任务与配置；不能直接推广到其他任务。每回合检查资产、锅位姿、操作点、TCP和夹爪初态，与冻结证明误差必须≤1e−8。

`run_research_jobs.py` 串行采集比较/确认各100个成功专家，再训练种子0、1各6000轮。采集每组最多500候选；训练单作业15000秒上限，120秒无进展发SIGUSR1、240秒无进展终止本任务进程组。GPU总预算172800秒，先前阶段以包含准备/空闲的保守墙钟上界计入；只有本任务作业被管理。

`run_learning_evaluation.py` 等待训练监督器通过后，验证六个权重有限、数据原件哈希不变、两种子划分和归一化一致，再启动六检查点×比较100回合及所选模型/同种子3k基线×独立确认100回合。若预算选择仍为3k，只运行一次确认并明确没有预算提升。每组真实评测后检查动作、原判据、TOPP调用、视频解码和逐种子结果；失败即停止并保留日志。比较结果、选择规则和确认结果写入checks，报告每分钟刷新。

有效命令与本机路径参数快照保存于 `experiments/configs/learning-study-20261106-225739/`。迁移/重跑须修改到新目录；已有目录均拒绝覆盖，训练权重不是包含优化器状态的精确续训断点。

```bash
source data/local-runs/fullchain-20261105-005503/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/xense/miniforge3/envs/robotwin/bin/python -u scripts/run_research_jobs.py --config /path/to/new-supervisor.json
/home/xense/miniforge3/envs/robotwin/bin/python -u scripts/run_learning_evaluation.py --run-dir /path/to/new-study
```

实际状态以新目录的 `supervisor/status.json`、`evaluation-pipeline/status.json` 和逐作业日志为准。总报告新增 `#planner-controls-20261106` 与 `#learning-study-20261106`，包含已完成证据及排队状态。长训练、模型比较和独立确认尚未完成前，不发布提升结论。

补充数值审计：`compare_policy_actions.py` 检查原生重复共同19种子、重置重复21种子的ACT逐动作数值和左右臂实际关节位置，所有最大差为0；这不意味着专家路径相同。真实冻结证明的反例检查也通过：训练/评测种子重叠、指定种子被替换、专家路径SHA变更都被拒绝，原始证明未修改。

比较清单已冻结：300000–300277共278候选，100条通过均为base0，144个base1候选全部不通过。独立确认清单继续从400000筛选，接纳数量和组成以运行状态为准。此资产覆盖局限在报告入口和清单表中直接披露。报告每分钟记录/home空间及GPU利用率/显存/功耗/温度，并在用户指定`#stack`入口链接最新阶段；结构检查与浏览器版面检查分别报告。

## 2026-11-07 worker故障诊断与新尝试

确认清单最终301候选/100通过，均base0；候选含160个base1，没有入选。首次6k训练在445个完整epoch、1336次更新后因worker SIGSEGV失败，训练监督器及排队评测均停止，未生成任何新检查点比较成绩。原日志、失败状态与清单完整保留于`learning-study-20261106-225739`。

诊断目录为`data/local-runs/worker-diagnostics-20261107-011118/`。50原始+50派生HDF5重新核对历史SHA全部通过；可用主存约87GiB、共享内存约62GiB，未发现耗尽证据。系统窗口未提供这次worker的内核崩溃条目；core限制为0，未安装coredumpctl。早前跨Python版本异常仍是线索，未证明硬件或某个库为根因；未改BIOS、驱动、全局环境或依赖版本。

`act_worker_diagnostics.py`在PyTorch初始化worker信号处理后重新注册faulthandler，按PID保存原生故障栈及最后一次getitem的回合/随机数状态，原子替换最后样本文件。两项回归验证了受控SIGSEGV能留下真实Python栈，以及观察器不改变worker的NumPy/Torch随机样本。故障注入测试的segmentation fault提示是有意产生的测试事件，不是正式训练成绩。

首个GPU外读取压力进程因新增长TMPDIR触发AF_UNIX path too long，未返回batch；保留原日志后仅终止该任务进程组，改用`data/tmp/`下较短的专用路径并重跑800轮/32000真实样本。新长作业在实际建立Unix socket成功后才进入loader，不用长度猜测替代真实检查；读取压力测试另有60秒batch超时与600秒总上限。这个IPC问题与原SIGSEGV分别记录，不互作未经证明的根因。

新训练目录`data/local-runs/learning-study-r2-20261107-011547/`复用已冻结的比较/确认清单，不重新筛选；种子0/1均重新初始化，不加载旧权重伪装精确续训。模型、数据、更新预算、spawn/持久worker及预登记选择规则保持，仅新增故障观察及IPC前置检查。

新增诊断的真实2轮/6更新短训练已完成，检查点、有限权重、worker复用及85字节实际socket路径检查通过。关闭观察器的真实短训练对照也完成；三次初始验证loss完全相同，但更新后权重两两最大差约4–4.8e−5，包括关闭观察器的情况。只能报告已观察到的小数值差异，不能声称位级一致或据此定位SIGSEGV根因。GPU外读取通过仍不能替代6k长训练或真实闭环评测。
