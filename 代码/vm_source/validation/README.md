# 具身智能线下营实训作业：证据索引

本目录只存放可复现的运行证据和检查脚本；原始 HDF5 数据集及模型检查点保留在项目对应目录。

## 第一项：H1 红杆传递数据集

- 任务：`Transmitting the red stick`
- 数据集：`../DataCollecter/dataset_transmitting_red_stick/`
- 结果：10 个 episode、每个 600 帧，共 6000 帧；`real_hdf5_validation.json` 中 `valid: true`。
- 数据结构说明：`hdf5_analysis_episode_0.txt`
- 采集日志：`h1_record_20260719-213303.log`
- 完整动作演示：`h1_red_stick_expert_demo_30fps_20s.mp4`（30 fps、600 帧、20 秒）。该文件由数据采集用的**专家轨迹**生成，画面已标注 `Expert`，用于清楚展示红棒由一只手交给另一只手的完整过程；不是ACT推理视频。

复查命令：

```bash
conda activate robot_sim
cd ~/unitree_h1_learn
python validation/validate_h1_hdf5.py DataCollecter/dataset_transmitting_red_stick
python DataCollecter/analyze_hdf5.py DataCollecter/dataset_transmitting_red_stick/episode_0.hdf5
```

## 第二项：ACT 训练与推理

- 使用模型：`../ACT/ckpt_models_medium_resnet34/policy_epoch_12000.ckpt`
- 训练配置：`../ACT/config.yaml`
- 任务和数据集均为红杆传递，不是 `Clean table`。
- 推理视频：`act_resnet34_epoch12000_final_30s.mp4`（20 fps、600 帧、30 秒）。
- 固定物体位置复查：`act_resnet34_epoch12000_fixedpose_30fps_20s.mp4`（30 fps、600 帧、20 秒，固定红棒位姿 `[0.40, 0.10, 1.03]`）。这是ACT模型推理视频。
- 传递动作提交视频：`act_resnet34_epoch12000_handover_to_end_30fps.mp4`（取完整 rollout 的第 400--599 步，即红棒已送至双手交接区域后的全过程到结束；30 fps、200 个原始顺序帧、6.667 秒）。
- 运行日志：`act_resnet34_epoch12000_final.log`
- 抽帧：`act_resnet34_epoch12000_final_20s.png`、`act_resnet34_epoch12000_final_29s.png`

推理代码将**模型输入**按配置中的 `camera_names = ['angle', 'top']` 排列；视频则固定显示为 `top | angle`，仅用于人工观察。此前二者混用会导致模型动作异常。

复查命令：

```bash
conda activate robot_sim
cd ~/unitree_h1_learn
export DISPLAY=:0
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
python ACT/demos/h1_act_eval.py --config ACT/config.yaml --epoch 12000 \
  --num-rollouts 1 --video-output validation/recheck.mp4 \
  --video-fps 20 --video-seconds 30 --force-exit
```

`--force-exit` 仅用于此虚拟机：其 MuJoCo/PyKDL 原生库会在 Python 销毁阶段崩溃，但在此之前视频已写完。该参数在视频写入完成并刷新输出后退出，不掩盖推理阶段的异常。

## 第三项：RoboJuDo G1 仿真

- BeyondMimic：`robojudo_g1_beyondmimic_bounded.log` 和 `robojudo_g1_beyondmimic_step90.png`。该测试使用 `g1_beyondmimic`，完成受控的 90 个仿真步。
- 策略切换：`robojudo_g1_switch_policy1.log` 和 `robojudo_g1_switch_policy1_step60.png`。日志显示从策略 0 切换到 `AMOPolicy`（策略 1），且 `active_policy_id=1`。
- 复现脚本：`run_robojudo_beyondmimic_evidence.py`。

`UnitreeCppEnv` 的导入检测仍是**外部依赖阻塞**：该虚拟机没有官方 `unitree_sdk2` 和由其编译的 `unitree_cpp` 扩展，详见 `robojudo_unitree_cpp_import.txt`。没有使用模拟模块替代该真实 SDK 检测。

## 总台账

机器可读的完整状态、文件路径和限制见 `assignment_evidence_manifest.json`。
