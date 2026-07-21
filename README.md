# Unitree Robot Exercise — 提交包

**生成日期：** 2026-07-21  
**用途：** 具身智能线下营 / 生产实习作业的可核查提交材料。  
**主仓库：** <https://github.com/haimichenha/unitree-robot-exercise>

## 目录与材料说明

| 目录 | 文件 | 作用与边界 |
| --- | --- | --- |
| `文档/` | `生产实习作业报告.docx` / `.pdf` | 最终报告的可编辑版与提交版 PDF；源模板未覆盖。 |
| `视频/` | `G1_行走-舞蹈策略切换_55秒_20fps.mp4` | G1 仿真策略切换证据：0–15 s 行走，15 s 切入舞蹈，40 s 返回行走，至 55 s 结束。 |
| `视频/` | `H1_ACT_fixedpose_模型推理_30秒_20fps.mp4` | ACT 固定物体位姿条件下的模型推理记录；仅说明该条件下的回放，不应表述为红棒交接成功。 |
| `视频/` | `H1_红棒传递_专家轨迹_20秒_30fps.mp4` | H1 红棒传递的专家采集轨迹，用于说明数据采集任务；不是 ACT 推理结果。 |
| `证据/` | `G1_策略切换关键帧.png` / `.log` | 关键帧和可解析运行日志；日志记录配置、帧数、键位映射、切换时刻及轨迹。 |
| `代码/` | `vm_source/` | 从虚拟机同步的完整实验代码包：ACT、H1 数据采集、MuJoCo/IK、验证及训练监控，共 50 个文件（源码、配置、XML 和说明）；不包含训练数据、检查点或外部项目克隆。 |
| `third_party/RoboJuDo/` | RoboJuDo `release@ed7601f` 快照

## G1 行走—舞蹈切换的可核查结论

- 使用配置：`g1_locomimic_beyondmimic`。
- 行走段使用前向速度命令 `0.55`；15.00 s 向现有 `KeyboardCtrl` 队列发送 `[` 的释放事件，触发 `[POLICY_MIMIC]` 和 `Dance_wose`；40.00 s 发送 `]` 的释放事件，触发 `[POLICY_LOCO]` 并恢复行走。
- 视频为离屏、可重复录制：它执行项目既有的键位映射分支，但**不是**桌面前人工手按键盘的屏幕录制。请以 `证据/G1_策略切换运行日志.log` 的事件时间、帧数（1100）和时长（55.000 s）复核。
- 视频 SHA-256：`D47E1499C287E635A984220B75944699329143F5AA9140A1EF9BD9DBBB4542C4`。

## 代码来源、复现与外部引用

代码从虚拟机中的 `~/unitree_h1_learn` 工作区同步。`代码/vm_source/` 保留完整的本次实验源码树，而非只保留 4 个脚本：

1. `ACT/`：训练、推理、DETR/VAE 网络、工具函数、12000 轮 ResNet-34 配置与 CPU 配置；
2. `DataCollecter/`：H1 采集、动作策略和 HDF5 检查；
3. `Mujoco_env/`：H1 MuJoCo 环境、IK、KDL 工具、机器人封装及必要 XML；
4. `validation/`：专家轨迹、HDF5、G1 舞蹈及行走—舞蹈切换的证据脚本；
5. `project_execution/scripts/`：训练监控脚本。

源码仅保留在 `代码/vm_source/`，便于直接浏览、运行和版本管理。`代码/代码说明.md` 给出目录级说明。

`代码/来源与边界.md` 明确区分了“本次提交的实验工程代码”和外部依赖：本仓库提交本次 H1/ACT/验证工程以及对 RoboJuDo 的调用与证据适配代码。

## 快速使用：G1 行走—舞蹈仿真

仓库已包含 RoboJuDo 固定版本，可直接阅读或在 Linux/WSL 中安装。建议为 RoboJuDo 使用独立 Python 3.11 环境，避免与 H1/ACT 的 Python 3.10 环境混用：

```bash
git clone https://github.com/haimichenha/unitree-robot-exercise.git
cd unitree-robot-exercise/third_party/RoboJuDo
conda create -n robojudo python=3.11 -y
conda activate robojudo
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python submodule_install.py       # 按 submodule_cfg.yaml 安装所选可选模块
python scripts/run_pipeline.py -c g1_locomimic_beyondmimic
```

在该配置的交互仿真中，`[` 切换到 MotionMimic（舞蹈），`]` 返回 LocoMotion（行走）。真实机器人 SDK、网络配置和附加模块不属于此“一键仿真”步骤；请先阅读 RoboJuDo 的本地 README 与许可证。

G1 策略切换依赖外部上游项目 **[HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo)**。本提交包记录的上游版本为 `ed7601f`。RoboJuDo 的源码、模型、资源和许可证文件**未被复制到本仓库**；复现实验时应自行从上游仓库获取，并遵守其许可证与依赖说明。上游 README 所示交互命令为：

```bash
python scripts/run_pipeline.py -c g1_locomimic_beyondmimic
# [ ：切换到 MotionMimic（舞蹈）
# ] ：切换到 LocoMotion（行走）
```

离屏复核脚本在已安装 RoboJuDo 及其依赖的环境中运行：

```bash
python validation/record_robojudo_g1_loco_dance_switch.py \
  --config g1_locomimic_beyondmimic --seconds 55 --walk-before 15 \
  --dance-seconds 25 --fps 20 --output validation/robot_switch.mp4
```

## 已知限制

- `UnitreeCppEnv` 的导入依赖 Unitree SDK2 与 `unitree_cpp` Python 绑定；当前虚拟机未安装这些外部依赖，因此本提交的 G1 证据来自 RoboJuDo/MuJoCo 路径，而非 `UnitreeCppEnv`。
- 12000 轮 ACT 检查点并未验证为“完整红棒交接成功”。故报告与文件名明确区分“模型推理记录”和“专家轨迹”，不将专家视频作为模型效果。
- 原始 HDF5 数据集、训练检查点、缓存、完整第三方仓库和大体积资源未纳入提交；代码包中的脚本用于定位、复核和复现流程。

## 完整性校验

执行以下命令核验文件：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\视频\G1_行走-舞蹈策略切换_55秒_20fps.mp4
Get-Content .\CHECKSUMS.sha256
```

`CHECKSUMS.sha256` 列出本项目代码、文档、视频和证据的 SHA-256。
