# RoboTwin 全流程运行交接

> 历史归档说明（2026-11-12）：以下正文保留 2026-11-04 至 2026-11-05 的阶段记录，
> 包含当时的限制、待确认项与机器状态，不代表当前授权或实验进度。
> 用户已要求不再以本文件作为执行依据；现行规则见 [AGENTS.md](AGENTS.md)，
> 后续复现结果及当前存储位置见 [实验记录](experiments/README.md)。
> 当前仅在 `test/robotwin-experiments` 分支提交、推送小型代码和文档；
> 不向 main 提交、不创建 PR，大型产物保留本地。

更新时间：2026-11-05。本文服务于从 `~/src/xensim-bench` 切换到 `~/src/RoboTwin`
后的开发会话。先读 [AGENTS.md](AGENTS.md)，再核对当前用户授权和本机状态。

## 1. 用户目标与背景

用户主要负责仿真资产搭建，熟悉自己的 xensim-bench，对 RoboTwin 尚不熟悉。
此前已完成 RoboTwin 代码调研与图文导读，现在希望在真实仿真中跑通：

```text
资产与配置
  → 创建 SAPIEN 场景
  → 运行脚本专家 / CuRobo 规划
  → 保留成功种子与双臂路径
  → 重建场景、回放录制 HDF5
  → 转换训练数据、训练 ACT
  → WebSocket 模型服务
  → 学习策略真实闭环评测
```

用户明确要求：

- 使用名为 `robotwin` 的 Mamba 虚拟环境管理。
- 先规划，用户确认方案后再执行；关键部分向用户汇报、确认。
- 不提交代码、不推送远端、不创建 PR。
- 规则与交接已建立；本会话已批准执行阶段 1–2，后续阶段仍需按节点确认。

## 2. 授权状态：全工程链路已授权，正式训练预算仍在节点 B 确认

用户在本会话只读复核并确认具体方案后，明确回复 **“开始执行！”**。
已批准：创建 Mamba `robotwin` 环境，按固定基线安装/编译依赖、初始化固定 gitlink
对应的 XPolicyLab、下载机器人与物体包、创建独立 `demo_clean` 副本，并完成
`lift_pot` / `aloha-agilex` 的 1 个专家成功 seed 与原生回放采集。
专家搜索上限为 20 个 seed 或 30 分钟（不含安装编译），连续三次相同异常停止。
完成单回合与三相机/数据验收后停在 **确认节点 A**，等待用户确认再扩大采集。

2026-11-05 用户回复“继续下一阶段的工作”，视为通过节点 A 并批准阶段 3：原任务/机器人/场景基线下，使用全新目录采集并验收 50 条成功示范。阶段 1–2 产物保留原样。

最新用户明确要求继续“数据回放、模型训练、模型推理评估等全链路”，已授权推进转换、短训练、checkpoint重载、服务、真实闭环评测及可复用工程验收；不止复现demo。正式长训练额度仍需在短训练实测后按节点B确认，建议1000 epoch尚未自动视为批准。

仍需确认：涉及动作对齐的转换/读取器语义修复具体差异；
另建环境、改系统工具链或基线版本、修改任务/控制/转换源码也需先给具体方案。
移除 MPLib 碰撞失败条件的补丁未获批准，未应用。常规安装与必要兼容处理
在已批准阶段内继续，记录原件、差异和验证；不重复要求相同授权。

仍禁止暂存、提交、推送或创建 PR，不操作 xensim-bench 源码与环境。
阶段 1–2 已验收完成：seed=0 首次专家尝试与回放成功，112 组 HDF5、三相机与视频通过检查；该阶段的任务均已退出；阶段 3 已全部完成且无活动作业，结果与限制见文末。

本轮进展、实际命令与失败记录见文末 2026-11-04 执行记录和
`data/local-runs/stage12-20261104-222636/`。

## 3. 代码与工作区基线（阶段 1 开始前快照）

- 本地根目录：`/home/xense/src/RoboTwin`，Linux 大小写敏感。
- 上游：`https://github.com/RoboTwin-Platform/RoboTwin.git`。
- 分支：`main`。
- HEAD：`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`。
- 提交时间：2026-09-24 11:44:58 +0800。
- 创建本次文档前，RoboTwin 工作区干净。
- XPolicyLab gitlink：`fa431ecd893ee706883e64fe5fe1464ec8cd928d`，当前未初始化。
- `assets/` 仅有下载脚本和示例图片；机器人、物体、纹理包尚未下载。
- 本会话未启动任何仿真、训练或模型服务，没有需要接管的后台作业。

`~/src/xensim-bench` 有大量其他未提交工作，不在本次运行范围内。
不要切换、重置或清理它，也不要复用其 Isaac/Python 环境来安装 RoboTwin。

## 4. 本机环境快照（安装前历史状态）

以下来自 2026-11-04 的只读检查，执行前必须刷新：

- Mamba：`/home/xense/miniforge3/bin/mamba`。
- `mamba env list` 尚无 `robotwin` 环境。
- GPU：一张 NVIDIA GeForce RTX 4090，24564 MiB，驱动 580.126.09。
  最近检查显存总占用约 1.3 GiB；更早的计算进程查询未发现训练或仿真作业。
- `/home` 所在分区约剩 97G；根分区只剩约 448M，`/tmp` 位于根分区。
- 系统 nvcc：`/usr/local/cuda-13.0/bin/nvcc`，CUDA 13.0。
- 系统 GCC/G++ 为 13；PATH 下没有 gcc-11/12 或 g++-11/12。
- `cmake` 在 PATH；`ninja`、`ffmpeg`、`vulkaninfo` 未在当前 PATH 找到。
  这不等于整个机器上没有相关库。
- 存在 `/usr/share/vulkan/icd.d/nvidia_icd.json`，但尚未执行 Vulkan 或 SAPIEN 出图验证。

### 安装规划的约束

官方安装页推荐 Python 3.10、CUDA 12.1；本地 requirements 固定了
SAPIEN 3.0.0b1、Torch 2.4.1、NumPy 1.26.4、MPLib 0.2.1 等；安装脚本固定
CuRobo v0.7.8，另外安装 warp-lang 1.12.0 与 setuptools 69.5.1。
这些是上游声明，不是本机已经验证兼容的环境。

系统 nvcc 13.0 不能直接视为匹配 Torch CUDA 运行库。优先在 `robotwin` 环境中准备
匹配工具链和编译器；不改变系统 CUDA、驱动或其他项目环境。
构建临时文件、pip/Hugging Face 缓存及 Torch 扩展编译目录应显式放在 `/home`。

## 5. 实验路线：阶段 1–3 已完成，阶段 4 待确认

首个任务建议使用 `lift_pot`：双臂抓锅的两个把手并抬升，步骤集中，适合展示双臂全链路。
它比从多类随机物体开始更容易定位工程问题，但不保证首次训练就成功。

- 任务：`lift_pot`。
- 基础场景配置：`demo_clean` 的独立实验副本，保留原任务随机范围与成功条件。
- 机器人：`aloha-agilex`，一体双臂。
- 观测：头部与左右腕 RGB，机器人状态；按官方 ACT 适配，不加入专家物体真值。
- 模型：ACT；joint/qpos 目标动作，双六轴加双夹爪，共十四维。
- 首轮专家验证：1 个真实成功回合。
- 数据规模：50 条成功自采示范；ACT 原生读取器为回合级 80/20 划分，
  需记录实际种子、划分及统计来源，不宣称已经形成或批准了该数据集。
- 训练：先短训练验证；正式首轮建议 1,000 epoch，预算需在短训练实测后再确认。
  官方脚本默认 6,000 epoch，不得未经确认直接长时间运行。
- 评测：先少量闭环检查，再用未参与采集的种子做建议的 20 回合；
  保留原任务阈值及官方专家筛选规则，并报告成功与失败。
- 首轮尽量在一个 `robotwin` 环境中完成，仿真和模型为不同进程；单卡采集、训练、
  评测分阶段执行。若需要另建策略环境，先解释依赖冲突并确认。

### 阶段 1：建立与验证环境

批准后创建环境，按固定版本手动梳理安装顺序，补齐构建和视频工具。
验证实际 Python 路径、依赖导入、CUDA 运算、CuRobo 编译与 SAPIEN 相机出图。
不要仅凭 `scripts/test_render.py` 打印 Render Well 就认定真实采图通过；它主要创建渲染器。

记录环境导出、实际包版本、安装日志、必要依赖补丁、源码/资源版本。
先核对磁盘容量，所有大文件避免经过根分区临时目录。

### 阶段 2：跑通一个专家成功回合

下载机器人和物体资源，维护独立实验名与新产物目录，执行原任务。
检查头部与双腕真实图像、双臂实际抓持、抬升和原 `check_success()` 返回值。
保留 seed、规划路径、配置、场景信息、HDF5 和视频。

**确认节点 A：**向用户展示场景、相机和专家成功回合；等待确认后再扩大采集。

### 阶段 3：采集和检查 50 条成功示范

走原生两阶段采集：成功种子/规划路径筛选，再相同 seed 重建回放录图。
正式采集使用新目录，不覆盖阶段 2 的单回合产物。启动前说明停止策略和失败监控。

抽查图像颜色、相机顺序、数据长度、状态/动作含义及视频对应关系；保留失败尝试。
采集目录和 ACT 转换器路径存在差异，按第 7 节建立可追溯路径映射，保留原始 HDF5。

### 阶段 4：ACT 数据转换与短训练

验证转换、实际划分、模型前向/反向、checkpoint 保存、重载和单步推理。
记录显存、实际每轮/更新耗时与短训练结果。不得把短训练完成或 loss 降低当作策略成功。

**确认节点 B：**向用户汇报数据质量、短训练和预计正式训练资源；确认预算后再长训练。
1,000 epoch 只是建议，不是已批准的运行额度；扩大到 6,000 epoch 需要另行确认。

### 阶段 5：模型服务与闭环评测

加载真实训练 checkpoint 及其归一化统计，经 WebSocket 连接 RoboTwin 客户端。
检查 reset、相机名、状态顺序、动作维度/单位、缓存与逐回合终止。
先少量检查，再按已批准方案评测未采集 seed。记录专家筛选、有效回合与跳过原因。
输出成功率、失败视频、推理/RPC/网络前向耗时及真实仿真执行情况。
不因模型失败修改任务阈值或初态分布。

### 阶段 6：交付复现入口与证据

给出环境、资源版本、实际执行命令、配置、数据清单、权重、服务命令、评测结果与回放。
区分环境就绪、专家成功、数据完整、训练完成、模型闭环成功及未完成项。
及时更新本交接文件，记录活动作业、日志、PID、当前阶段和下一确认节点。

### 需要单独确认的变化

- 修改任务、机器人控制或数据转换源码：先提供具体差异、必要性与验证方案。
- 改成功判据、初态分布、动作/观测契约、专家筛选或基线版本。
- 另建环境、改系统工具链、下载第一轮不需要的大资源、明显扩大训练/评测预算。
- 不要把这些节点扩展成对已批准范围内每条常规命令都重复询问。

## 6. 资源下载与存储规划

已查询官方 Hugging Face 元数据，以下为当时压缩包体积，不包含解压、缓存和训练产物：

- `embodiments.zip`：219,847,741 bytes，约 0.20 GiB。
- `objects.zip`：3,737,778,549 bytes，约 3.48 GiB。
- `background_texture.zip`：10,970,687,027 bytes，约 10.22 GiB。

干净场景第一轮建议只下载机器人和物体；不开启随机背景/干扰物时，暂不下载纹理大包。
执行前检查实际依赖和下载 revision；若后续确需纹理，再汇报体积与用途。
压缩包大小不能当作最终空间预算，解压和包安装前重新检查剩余空间。

建议每轮使用 `data/local-runs/<run-id>/` 管理日志、配置快照、缓存和结果索引，
每阶段独立新目录。采集器会在配置的 `save_path` 下继续追加
`<task_config>/<task>/<embodiment>/data/episode_0000000.hdf5`，需记录最终真实路径。
上游 ACT 的 processed_data、checkpoints、TASK_CONFIGS.json 等固定路径单独核对，
不能假定都能自动改到统一目录。

## 7. 已发现的关键实现细节

### 不直接运行上游一键安装脚本

`scripts/_install.sh` 会调用 `scripts/update_xpolicylab.sh` 拉取配置分支的最新提交，
可能偏离 gitlink 固定版本。更新脚本的 `--stage` 还会暂存子模块指针，当前禁止使用。
需在获批后逐项安装并核对固定版本，避免无意升级。
安装脚本还修改 SAPIEN URDFLoader 和 MPLib 的部分实现；这些补丁需留下可检查差异。

### 采集与 ACT 转换目录不相同

当前原生采集保存到：

```text
data/<task_config>/<task>/<embodiment>/data/episode_0000000.hdf5
```

锁定 XPolicyLab 的 `policy/ACT/detr/process_data.py` 则按策略目录的相对路径读取：

```text
../../../data/<bench_name>/<ckpt_name>/<env_cfg_type>/data/
```

默认机器人采集目录名是 `aloha_agilex`；XPolicyLab 的动作 profile 常用 `arx_x5`。
`embodiment` 是仿真实体，`env_cfg_type` 是状态/动作 profile，需校验维度和语义。
优先考虑明确的数据路径映射或现有参数，不复制主流程、不改写原始 HDF5。
若必须改转换源码，先给用户看具体差异并确认。路径衔接尚未实际验证。

### 观测、动作、时间与颜色

- 原生 `get_left/right_arm_jointState()` 读取 drive target，而非实测 qpos。
  转换器用相邻采样帧构造 state[t] 与 action[t]=目标[t+1]，不能把命令字段当实测响应。
- `save_freq=15` 是中段每十五个物理步记录一次，不是严格 15 Hz；默认 250 Hz 下
  中段约 16.7 Hz，原语边界还有额外采样。HDF5 的 frequency 元数据直接写 15。
- 存储相机字节使用 `decode_image_bit()` 解码、`encode_image_bit()` 编码。
  历史/标准 JPEG 字节格式共存，不能在正确解码后再统一交换 RGB/BGR。
- 内参、CV 外参和 GL cam2world 不可混用；当前采集与评测同名矩阵字段的优先来源不同。
- `load_robot()` 会把机器人 link 的质量设为 1；不能把原 URDF 声明等同于运行值。
- 模型 joint 动作经 TOPP、ee 动作经规划后推进多个物理步；一次 RPC 不等于固定周期物理步。

### ACT 和评测

- 固定版本 ACT 需要 checkpoint 目录内的 `policy_last.ckpt` 和 `dataset_stats.pkl`。
- 适配器按相机名读已解码 RGB，并缩放到 640×480；不能擅自只在推理端降尺寸。
- 官方默认 chunk_size=50、batch_size=16、lr=1e-5、num_epochs=6000；
  这些是默认预算，不是当前用户已批准的运行参数。
- 网络预测动作段，ACT 底层 get_action 每次选择当前动作；缓存、时间聚合和模型 reset 要核对。
- 当前远端评测默认 expert_check=True，先过滤专家不能完成的 seed 再重建评测。
  记录筛选条件与跳过原因，避免把该成功率解释成全部随机初态上的结果。
- `scripts/collect_data.py`、`scripts/eval_policy.sh` 是当前入口；旧版本教程的目录可能不适用。
- `.gitignore` 虽写了 `envs/robot/planner.py`，该文件仍已被 Git 跟踪。
  不要据 ignore 文本判断任何源码都可随意修改或无需核对差异。

## 8. 已有调研资料与源码入口

图文导读（26 幅图表、官方标注截图、相机覆盖示意、xensim 升级建议）：

[打开本机 HTML 导读](../xensim-bench/outputs/research/robotwin-overview-20261104/robotwin.html)

报告不是本机物理成功证据；本地路径只适用于当前机器。可独立重查的来源：

- [RoboTwin 官方安装](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- [官方物体标注](https://robotwin-platform.github.io/doc/usage/object-annotation.html)
- [官方机器人接入](https://robotwin-platform.github.io/doc/usage/new-embodiment.html)
- [官方资产库](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main)
- [固定版本 XPolicyLab ACT](https://github.com/XPolicyLab/XPolicyLab/tree/fa431ecd893ee706883e64fe5fe1464ec8cd928d/policy/ACT)

本地建议阅读顺序：

1. [lift_pot](envs/lift_pot.py)：双臂专家与成功条件。
2. [配置](env_cfg/task_config/demo_clean.yml)、[基类](envs/_base_task.py)：初始化与执行。
3. [机器人](envs/robot/robot.py)、[规划器](envs/robot/planner.py)：坐标与轨迹。
4. [采集](scripts/collect_data.py)、[HDF5 转换](envs/utils/pkl2hdf5.py)。
5. [评测入口](scripts/eval_policy.sh)、[客户端](scripts/eval_policy_xpolicylab.py)。
6. 子模块初始化后，读取其规则及 `policy/ACT/README.md`、`detr/process_data.py`、
   `train.sh`、`model.py`、`detr/act_policy.py`。

## 9. 新会话第一步：只读复核

在 RoboTwin 根目录读取规则与交接，再执行以下只读命令；它们不是安装/运行授权：

```bash
pwd
git branch --show-current
git status --short
git log -1 --format='%H %ci %s'
git submodule status
mamba env list
df -h /home/xense/src/RoboTwin /tmp
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
```

随后汇报实际差异，以第 2 节最新授权和文末验收结果为准确定下一阶段。
阶段 1–3 已完成，不要因新会话重复安装或采集；当前阶段 4（ACT 转换与短训练）待用户确认。
若用户已明确批准当前下一阶段，则在该范围内推进，不重复索要相同授权；不要把打开目录解释成全部长任务批准。

## 10. 后续更新记录格式

```text
更新日期：
当前阶段与用户已批准范围：
基线 / 环境 / 配置：
实际执行命令：
产物与日志路径：
实际完成和验收证据：
失败、限制与未执行检查：
活动作业（PID、启动方式、如何停止）：
下一步及是否需要用户确认：
```

文档创建时状态（历史）：规则与交接文件已创建；当时执行计划待确认，没有运行作业。当前状态见第 2 节及文末执行记录。

## 2026-11-04 阶段 1–2 已批准，执行中

用户在复核具体方案后明确回复“开始执行！”。已批准创建 `robotwin`、固定版本安装/编译、初始化 gitlink 对应的 XPolicyLab、下载机器人与物体包，并完成 `lift_pot` 单成功 seed 及原生回放采集。专家搜索上限 20 个 seed 或 30 分钟（不含编译），完成后停在确认节点 A。未批准扩大采集、训练或评测；不暂存/提交/推送/建 PR。涉及移除 MPLib 碰撞失败条件的补丁仍需单列确认，暂不应用。

运行目录：`data/local-runs/stage12-20261104-222636/`，包含 env.sh、logs、snapshots、checks 与 status.json。临时/下载/编译缓存全部放 /home；初始 /home 约剩 91G，根分区约剩 441M。基线不变，环境和资产尚未建立，开始创建环境。默认 exec 沙箱报 bwrap 网络命名空间错误，因此已批准操作通过 require_escalated 执行。

阶段 1 进展：`mamba create -n robotwin -c conda-forge --strict-channel-priority python=3.10 pip setuptools=69.5.1 wheel ninja cmake ffmpeg gcc_linux-64=12 gxx_linux-64=12 -y` 已成功；Python 实际为 3.10.22，GCC 为环境内 12.4。`mamba install -n robotwin -c nvidia/label/cuda-12.1.1 -c conda-forge cuda-toolkit=12.1 -y` 已成功，nvcc 12.1.105。Torch 2.4.1+cu121 / torchvision 0.19.1+cu121 安装成功；其临时拉入的 NumPy 2.2.6 正由后续固定约束安装替换为 1.26.4，尚未做仿真导入验证。

`git submodule update --init --recursive XPolicyLab` 完成，子模块干净；CuRobo v0.7.8 对应 d64c4b005459db10c5dd867d8b30a87d5bda9bdb 已下载，尚未编译。资产 revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186，两个 ZIP 均对照官方 LFS SHA256 校验成功，解压共约 5.2 GiB；已运行上游 `scripts/update_embodiment_config_path.py` 生成下载资源内的本机路径配置，未改任务源码。原始 ZIP、官方 metadata 和文件清单保存在运行目录 sources/ 与 snapshots/。

当前安装命令：`source <run>/env.sh; mamba run -n robotwin bash <run>/install_python.sh`，日志 `logs/pip-base.log`；完成后需检查 `pip-base.exit`，再运行 `build_curobo.sh` 和 `check_environment.py`。还未启动仿真、专家、训练或服务。其他本次安装与下载会话均已结束，详细退出码在 logs/*.exit。

阶段 1 继续：固定 Python 依赖与 XPolicyLab 可编辑安装完成，`logs/pip-base.exit=0`、pip check 无依赖错误；NumPy 现为 1.26.4。`mamba run -n robotwin bash <run>/build_curobo.sh` 正编译 CuRobo，日志 `logs/curobo-build.log`，会写同名 .exit。

真实 SAPIEN RT 相机已通过：`source <run>/env.sh; mamba run -n robotwin python <run>/check_render.py`，退出码 0，320×240 RGB，`checks/sapien-render.png` / `checks/render.json`。第一次验收脚本用了旧 color 参数导致 TypeError，按已安装 SAPIEN 3 的 RenderMaterial 接口修正本机检查脚本后通过，首次日志 `logs/render-check-attempt1.log` 保留；不是修改运行依赖或任务源码。到目前尚无上游依赖补丁，MPLib 碰撞判断未改，任务/控制/转换源码未改。

独立配置 `env_cfg/task_config/local_lift_pot_stage2_20261104_222636.yml` 已生成，完整复制 demo_clean，仅 episode_num=1、save_path 指向运行目录 expert/。待环境验证通过后执行 `source <run>/env.sh; mamba run -n robotwin python -u <run>/supervise_collect.py`；监督脚本只调用原生 scripts/collect_data.py，记录 PID/进程组及 20 seed / 1800 秒上限、连续三次相同异常停止，不复制专家/采集主循环。专家尚未启动。

CuRobo 首次构建失败：前四组扩展通过，geom 的 sphere_obb_kernel.cu 在 cuda_bf16.hpp 报 `_NV_EVAL1` 宏参数未结束。原日志/退出码保留为 `logs/curobo-build-attempt1.*`。以相同源码、CUDA 12.1/GCC 12.4/架构 8.9 单独编译该文件，仅将 nvcc --threads=8 改为 --threads=1 后退出码 0，生成 checks/sphere.o；复现命令与日志保存在 snapshots/sphere-build-command.txt、logs/cuda-repro.*。完整重构建使用 `NVCC_APPEND_FLAGS=--threads=1` 覆盖内部线程数，未修改 CuRobo/系统/任务源码或版本。CUDA 真实矩阵运算已通过。

阶段 1 已通过：CuRobo 全部五组扩展构建成功，logs/curobo-build.exit=0，综合环境检查 logs/environment-check.exit=0；checks/environment.json 记录固定版本、全部必需模块导入、真实 CUDA 运算与已验收 RT 图像。最终 pip check 无依赖错误；snapshots/ 保存 environment.yml、conda-explicit.txt、pip-freeze.txt。无需修改任何上游依赖源码；唯一构建选项调整为 NVCC_APPEND_FLAGS=--threads=1，已加入本次 env.sh。可选点云依赖 pytorch3d 未安装，本次 RGB 配置不调用其 FPS 路径。

阶段 2 已启动：`source <run>/env.sh; mamba run -n robotwin python -u <run>/supervise_collect.py`。原生子进程命令、PID/PGID 在 logs/expert-process.json，完整输出 logs/expert-collect.log，监督日志 logs/expert-supervisor.log。停止仅对该 JSON 中本任务进程组发 SIGINT；监督器会执行 20 seed / 1800 秒和重复异常限额。实际输出根目录 expert/local_lift_pot_stage2_20261104_222636/lift_pot/aloha_agilex/。成功后必须执行 check_episode.py、查看三相机与视频，再到确认节点 A 等待。

## 阶段 1–2 最终结果：等待用户确认节点 A

`lift_pot` 首次专家尝试 seed=0 成功，0 次失败，规划与回放总耗时约 31.6 秒，原生进程退出码 0、stop_reason=null，回放后原成功断言通过。未修改任务阈值、初态分布、动作/观测契约、筛选规则或源码。

数据验收命令：`source data/local-runs/stage12-20261104-222636/env.sh; mamba run -n robotwin python -u "$ROBOTWIN_RUN/check_episode.py"`。退出码 0；112 组 state/action/三相机样本，三相机均 320×240 RGB，全部 336 帧图像经共享 decode_image_bit 解码；状态、动作、内外参全部有限且长度对齐，关节维度 6+1+6+1，相邻 state/action 移位关系通过。原视频 113 帧（30 FPS），比配对 HDF5 多末观测一帧，符合当前上游转换逻辑。注意 state 是 drive target 而非实测 qpos；frequency=15 是 save_freq 的元数据，不代表严格 15 Hz。

本轮产物均在 `data/local-runs/stage12-20261104-222636/`：
- `RESULT.md`、`status.json`：结果与状态；`checks/environment.json`、`checks/episode.json`：验收数据。
- `checks/three-cameras.mp4`：同步三相机派生预览；`checks/three-cameras.jpg`：首/中/末帧，已查看确认双臂抓持及抬升。
- `expert/local_lift_pot_stage2_20261104_222636/lift_pot/aloha_agilex/`：seed.txt、_traj_data/episode0.pkl（每侧三段路径）、scene_info.json、instruction/、data/episode_0000000.hdf5 与原生 video/episode_0000000.mp4。
- `snapshots/`：精确源码版本、资源 revision/LFS SHA256/原始数据卡、配置、pip freeze、Mamba 导出。
- `logs/`：安装/构建/渲染/专家/数据检查日志及退出码，保留两次失败诊断记录。

最终 Git 检查：main/HEAD 不变，无已跟踪文件或暂存改动；XPolicyLab 固定提交且干净，CuRobo 原始源码干净。本次新增独立配置以及更新本地 HANDOFF，AGENTS/CLAUDE 保持原样。未暂存/提交/推送/建 PR。

原生专家 PID/PGID 3443215 已退出；进程组剩余进程为空，见 checks/active-processes-final.json；无本任务活动作业。最终 /home 约剩 57G，根分区约剩 410M，GPU 显存约 1.1 GiB。下一步：向用户展示结果并等待确认节点 A；未经确认不启动 50 条正式采集、训练或评测，新采集须新配置/新产物目录。

## 2026-11-05 阶段 3 开始

授权：用户“继续下一阶段的工作”，批准原定下一阶段 50 条成功示范采集及质量验收；未扩大到转换、训练或评测。复核 main/HEAD、固定子模块和 robotwin 环境未变，无已跟踪文件改动。GPU 桌面/远程桌面占用约 1.6 GiB，/home 约剩 57G、根分区约剩 422M。

新运行目录 `data/local-runs/stage3-20261105-001152/`，配置 `local_lift_pot_50_20261105_001152.yml` 完整复制 demo_clean，仅 save_path 指向本次 raw/，原配置 episode_num 已为 50，不改随机范围、成功阈值、动作或观测。新产物根目录 `raw/local_lift_pot_50_20261105_001152/lift_pot/aloha_agilex/`。单回合实测约 8 MiB，50 条预计原始产物约 0.4 GiB，另为逐回合临时图像及派生检查留空间；所有缓存/TMPDIR 在 /home。

停止策略已在启动前告知用户：最多 200 个专家 seed 或总计 60 分钟；连续 5 次专家失败、3 次同类异常、600 秒无输出、/home 空间低于 20 GiB、根分区低于 200 MiB 时停本任务进程组。原生筛选与回放断言不变，不静默剔除回放失败。所有成功/失败 seed 与日志保留。未来评测须排除所有实际采集 seed；当前官方评测初始 seed=100000*(1+训练seed)，与本轮 0..199 搜索范围分离，实际评测待批准。

阶段 3 验收含所有回合有限数值、14 维 state/action、相邻移位/相机对齐、全部三相机图像共享解码、原视频对应关系与 SHA256 清单，抽查首/中/末帧和不同物体实例。ACT 数据路径/profile 映射先记录清单，不修改转换器或原始 HDF5，不在本轮生成训练划分/归一化统计或启动训练。

正式采集已启动：`source data/local-runs/stage3-20261105-001152/env.sh; mamba run -n robotwin python -u "$ROBOTWIN_RUN/supervise_collect.py"`。原生 PID/PGID 3503091，监督器 PID 3502750，命令和限额在 logs/process.json；完整日志 logs/collect.log，简要进度 logs/supervisor.log，定期统计 checks/progress.json 与 status.json。只对该任务进程组发 SIGINT 停止，不操作其他进程。结束后必须检查 supervisor.exit 与 checks/process-result.json；成功后运行本次 check_dataset.py 并查看抽样三相机证据。

阶段 3 首次监督运行提前停止（尚未回放）：63 次尝试、31 个成功、32 个失败，约 108 秒。停止原因是本机监督脚本错误地把间隔成功回合的三条同类 Error 累计为“连续三次”。此为监控计数错误，不是任务环境失败率阈值，也未修改原生结果。原件归档 logs/*-attempt1.*、checks/process-result-attempt1.json、snapshots/supervise_collect-attempt1.py。

已修正监控为三个连续专家尝试均出现同一 Error 才触发（成功或无此异常的失败会断开序列）；再次启动时保留完整首次 attempts/errors 并累计总 200 次尝试及自首次启动起总 60 分钟预算。原生 seed.txt 有 31 个成功 seed、规划路径完备，没有 HDF5；使用 `supervise_collect.py --resume` 调用原生断点，下一 seed 从最后成功 60+1 开始，故可能重复尝试此前失败的 61/62，日志会如实记录运行序号，不删旧失败。原任务源码、筛选、分布与成功阈值不变。

修正后的原生续采 PID/PGID 3520600，监督器 PID 3520253，当前日志仍为 logs/collect.log 与 logs/supervisor.log，首次日志保存在带 attempt1 后缀文件。预算起点 Unix 1793808830.7614565，状态含两次运行累计尝试，未放宽原生筛选。

第二次停止后的断点完整性检查发现原生 seed.txt 在被 SIGINT 打断的覆盖写中只留 27 个前缀 seed，虽然成功日志和规划文件有 37 条。第三次启动被预检断言拦下，未启动仿真。中断原件已保存 snapshots/seed.interrupted-attempt2.txt；逐条验证 37 条成功日志 episode_index=0..36、唯一 seed、每条左右三段规划文件存在且保存 SHA256，确认原文件字节精确等于成功清单前 27 条后，恢复完整工作断点 seed.txt。恢复证据 checks/seed-recovery.json，未改任何专家成功/失败标签，尚无 HDF5，之后仍须全部原生回放断言通过。

普通专家连续失败阈值调整为 10 已先向用户说明，仍受原累计 200 次尝试与首次启动起 60 分钟预算约束；三个连续同类异常与空间限额不变。第三次有效续采运行序号 3，从已证明的最后成功 seed75+1 继续。第一次尝试修复时 shell 未 set -e 导致预检失败后仍调用监督器，但监督器自身断言立即拒绝启动子进程；日志保存在 logs/resume-preflight-failed.*。现使用 set -e。

第三次有效续采 PID/PGID 3540092、监督器 3539751，run_index=3。停止仅发信号给此进程组，详见 logs/process.json。

50 个成功 seed 已筛满，最后成功 seed=111，正在原生同 seed/路径回放采图。累计历史包含断点重试，最终唯一 seed 数与重试次数由全量验收统计，不把重试去重后伪装成未失败。原成功筛选和回放断言均保持原样。全量数据检查已排队，命令 `mamba run -n robotwin python -u "$ROBOTWIN_RUN/check_dataset.py"`，只有 supervisor.exit=0 才运行；检查记录 logs/dataset-check.*、checks/manifest.json 与 checks/dataset-summary.json。

50 条原生回放已全部完成：累计 119 次尝试（112 个唯一 seed，7 次断点重试），50 成功、69 次失败尝试，最终 seed=111；第三次原生进程退出码 0，stop_reason=null，总耗时约 886.6 秒（含两次诊断间隔），50 个回放成功断言全部通过。

首次全量验收在检查脚本“vision 恰好三相机”的错误假设处失败；只读核对源码和阶段 2 原文件确认原生还始终采集机器人 static_camera_list 中的 front_camera，映射为 cam_third_view。配置 third_view=false 控制的是另一条 observer 路径，并不关闭这个静态 front 流。阶段 2 数据早已同样包含此第四路，原文件哈希不变，之前的336帧检查/预览指ACT所需三路，非全部原生相机。现修正本机检查脚本为验证全部四路（头+双腕+静态front），ACT输入仍明确只选头+双腕，原始数据、相机设置及上游源码不改。失败检查原件在 snapshots/check_dataset-attempt1.py、logs/dataset-check-attempt1.log。

## 2026-11-05 阶段 3 最终交接

目标与授权：用户“继续下一阶段的工作”批准50条成功采集及验收，已完成；尚未批准ACT转换/短训练，未运行正式训练、服务或评测。

结果：112个唯一候选seed（0..111）中50个成功、62个未通过；包含7次断点重试共119次尝试、69次失败尝试。最终原生退出码0，50条回放check_success断言全部通过。全数据集验收与来源检查退出码均0：5,553组对齐样本、106..127帧/回合、中位111；14维state/action及相邻移位、内外参、有限数值、所有视频长度均通过。原生四路（头/双腕/静态front）共22,212图逐帧共享解码；ACT仍只用头+双腕3路（16,659图）。HDF5合计404,384,707 bytes；详细SHA256和seed/路径在checks/manifest.json、metadata-sha256.json。

重要覆盖限制：50条成功场景全为060_kitchenpot/base0，base1没有进入成功示范集；未为补齐模型类别改分布/成功阈值/专家筛选。失败seed对应模型没有原生记录，不能直接断言全部失败均为base1。后续训练/评测解释必须保留此限制。

产物与入口：data/local-runs/stage3-20261105-001152/RESULT.md、status.json、checks/dataset-summary.json、checks/provenance.json；raw/local_lift_pot_50_20261105_001152/lift_pot/aloha_agilex/下有seed、每侧3段轨迹、场景、指令、HDF5和视频。checks/episode_0000000-three-cameras.mp4与episode_0000049-three-cameras.mp4为派生同步预览；已查看ep0/29/49首中末帧，抓持和抬升可见。全部实际命令、失败和监控/断点恢复证据见RESULT.md及logs/，stage2原HDF5哈希不变，37条恢复轨迹哈希不变。

ACT衔接：data/RoboTwin/lift_pot_local50_20261105_001152/arx_x5为新符号链接，指向原始产物根目录，50个HDF5均samefile；映射详见ACT_PATH_MAPPING.md、checks/act-path-mapping.json。未执行转换，未生成processed_data、实际划分或归一化统计。后续官方转换640×480三路RGB预计约14.30 GiB，仅是图像数组估算。官方读取器80/20为40/10，默认统计源为全部回合且末帧补齐，需在下一阶段记录实际索引与统计来源，不宣称train-only统计。

基线与改动：main/HEAD、XPolicyLab、CuRobo、环境版本未变；Git无已跟踪或暂存差异，子模块和CuRobo源码干净。本阶段仅新增独立配置（只改save_path）、运行目录/别名与更新HANDOFF。未暂存/提交/推送/建PR。

活动作业：三次原生进程组3503091/3520600/3540092均已退出，来源复核证实无剩余进程。下一步：向用户汇报数据结果及base0覆盖限制，等待阶段4（ACT转换与短训练）授权；正式训练预算仍在节点B确认。

## 2026-11-05 全链路任务启动

用户明确扩展范围为验证工程完整链路（回放、训练、推理、真实评测），已授权阶段4及后续工程集成验证；正式长训练预算仍按节点B，先短训练实测后提出具体参数。新运行目录 `data/local-runs/fullchain-20261105-005503/`，不覆盖stage3数据或旧产物，仍使用robotwin，GPU/磁盘分阶段执行。当前 /home约剩57G、根分区约407M、GPU桌面约1.6GiB，Git与子模块无已跟踪改动。

发现真实数据对齐兼容问题：RoboTwin原始HDF5已是state[t]/下一目标action[t]，ACT读取器is_sim=None又从action[k-1]起步。最小补丁提案在 reviews/act-action-alignment.patch / .md，尚未应用，需用户确认：只为RoboTwin派生数据加action_alignment标记、让读取器使用现有无额外错移分支；原数据和其他来源不变。待批准前可继续依赖和服务入口检查，不做依赖此语义选择的转换/训练。

全链路预备进展（等待动作对齐修复批准）：缺失的einops==0.8.1及本地DETR已在robotwin安装，固定Torch/NumPy等基线不变，pip check通过；官方ResNet18权重已缓存，SHA256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec，详情checks/backbone-weights.json。

动作错移已用真实episode0和未修改的上游转换/读取器复现：采样k时首动作等于原始action[k-1]而非action[k]，最大分量误差0.1066766，见reviews/alignment-reproduction.json；该诊断只把真实单回合派生写入reviews/alignment-baseline-data，不是正式50回合转换，更不是模型训练或闭环验收。已通过request_user_input_async请求批准最小补丁，尚未收到回答，未应用XPolicyLab源码修改。

已新增可复用入口scripts/train_act_recorded.py和scripts/eval_act_recorded.py：分别调用上游训练main/评测main，通过只读观察记录真实划分、统计来源、loss/梯度/显存/checkpoint校验，以及候选seed/模型实例/专家筛选、RPC与模型方法耗时、真实机器人动作和逐回合成功；不复制训练或评测主循环、不更改控制/成功判据。当前仅做语法检查，未执行这些新入口，不宣称其功能验收已通过。

配置位于fullchain运行目录：short-training.json为10 epoch、batch16、lr1e-5、chunk50、hidden512、dim_feedforward3200、seed0；正式1000epoch/save500仅存为formal-training-proposal.json，仍需短训练实测后节点B批准。独立完整评测配置local_lift_pot_eval_20261105_005503.yml只改save_path/collect_data；server-short.yml由官方eval_policy_server.py --dry-run通过，loopback19140，仅预演未启动。eval-short.json为2个真实回合接线验收，eval-formal-proposal.json为20回合；模型训练/服务/真实推理评估尚未执行。

当前无活动作业。下一步必须先等用户对reviews/act-action-alignment.patch明确答复；批准后应用并保留子模块diff，调用官方process_data.sh转换50回合，验证实际读取器k>0与原始action[k]一致，再执行短训练、checkpoint重载及RPC/少量真实仿真。实测后按节点B提出正式训练时间/显存与预算确认；继续完整链路任务，不把短训练或loss下降等同策略成功。
