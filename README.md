# Unitree Robot Exercise — 提交包

**生成日期：** 2026-07-21  
**用途：** 具身智能线下营 / 生产实习作业的可核查提交材料。  
**主仓库：** <https://github.com/haimichenha/unitree-robot-exercise>

## 目录与材料说明

| 目录 | 文件 | 作用与边界 |
| --- | --- | --- |
| `文档/` | `生产实习作业报告_具身智能线下营.docx` / `.pdf` | 最终报告的可编辑版与提交版 PDF；源模板未覆盖。 |
| `视频/` | `G1_行走-舞蹈策略切换_55秒_20fps.mp4` | G1 仿真策略切换证据：0–15 s 行走，15 s 切入舞蹈，40 s 返回行走，至 55 s 结束。 |
| `视频/` | `H1_ACT_fixedpose_模型推理_30秒_20fps.mp4` | ACT 固定物体位姿条件下的模型推理记录；仅说明该条件下的回放，不应表述为红棒交接成功。 |
| `视频/` | `H1_红棒传递_专家轨迹_20秒_30fps.mp4` | H1 红棒传递的专家采集轨迹，用于说明数据采集任务；不是 ACT 推理结果。 |
| `证据/` | `G1_策略切换关键帧.png` / `.log` | 关键帧和可解析运行日志；日志记录配置、帧数、键位映射、切换时刻及轨迹。 |
| `代码/` | `vm_source/` 与 `unitree_robot_exercise_code_20260721.zip` | 从虚拟机工作区逐项同步的自编/适配脚本和 CPU 配置；不包含第三方上游源码、模型权重或数据集。 |

## G1 行走—舞蹈切换的可核查结论

- 使用配置：`g1_locomimic_beyondmimic`。
- 行走段使用前向速度命令 `0.55`；15.00 s 向现有 `KeyboardCtrl` 队列发送 `[` 的释放事件，触发 `[POLICY_MIMIC]` 和 `Dance_wose`；40.00 s 发送 `]` 的释放事件，触发 `[POLICY_LOCO]` 并恢复行走。
- 视频为离屏、可重复录制：它执行项目既有的键位映射分支，但**不是**桌面前人工手按键盘的屏幕录制。请以 `证据/G1_策略切换运行日志.log` 的事件时间、帧数（1100）和时长（55.000 s）复核。
- 视频 SHA-256：`D47E1499C287E635A984220B75944699329143F5AA9140A1EF9BD9DBBB4542C4`。

## 代码来源、复现与外部引用

代码从虚拟机中的 `~/unitree_h1_learn` 工作区同步，保留的仅是本次实验所需脚本：

1. `record_robojudo_g1_loco_dance_switch.py`：固定时长离屏录制并验证策略切换；
2. `run_robojudo_beyondmimic_evidence.py`：有界步数运行并导出 PNG 证据；
3. `analyze_hdf5.py`：检查 HDF5 采集数据的结构；
4. `config.cpu.yaml`：无 CUDA 的虚拟机 CPU 训练配置。

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

所有提交文件的 SHA-256 均在 `CHECKSUMS.sha256` 中列出。
