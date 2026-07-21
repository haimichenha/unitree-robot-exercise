import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import sys
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
sys.path.append(HOME_PATH+'/ACT')

from utils.utils import set_seed
from utils.io_utils import IOUtils
from utils.model_interface import ModelInterface
import random
import time
from torchvision import transforms
import cv2
import sys
import torch
import time
import argparse
from pathlib import Path

sys.path.append(HOME_PATH)
from Mujoco_env.envs.h1_ik import make_sim_env

from DataCollecter.h1_record import sample_transfer_pose


R2D=180/3.1415926


class H1DexEnvInference:
    def __init__(self, mujoco_env,
                 peg_pose,
                 ):
        self.mujoco_env=mujoco_env
        self.peg_pose=peg_pose
        
    def get_image(self,obs):
        top=obs['images']["top"]
        angle=obs['images']["angle"]

        target_size = (320,240)  # Replace with your desired dimensions
        
        top = cv2.resize(top, target_size, interpolation=cv2.INTER_LINEAR)
        angle = cv2.resize(angle, target_size, interpolation=cv2.INTER_LINEAR)
        
        self.view_flag=True
        # Preserve the named camera frames here.  The policy input order is
        # defined by config['camera_names']; it is not necessarily the same
        # as the convenient top/angle order used for the evidence video.
        return {"top": top, "angle": angle}
    
    def write_img(self,img1,img2):
        pass
        output_image_path = "output_image.jpg"  # 输出文件的路径
        cv2.imwrite(output_image_path, np.hstack((img1,img2)))

    def get_state(self):
        obs=self.mujoco_env._get_qpos_obs()['qpos']
        return obs
    
    def step(self,action):

        action[:]=action[:]/R2D
        self.mujoco_env.step_all_simple(action)

    def reset(self):
        self.mujoco_env.reset(self.peg_pose)
        self.mujoco_env.render()
        time.sleep(1)
        
def get_image(img1, img2, device):
    # img1=img1[:, :, ::-1]
    # img2=img2[:, :, ::-1]
    # output_image_path = "output_image.jpg"  # 输出文件的路径
    # cv2.imwrite(output_image_path, np.hstack((img1,img2)))
    curr_images = []
    img1=rearrange(img1, 'h w c -> c h w')
    img2=rearrange(img2, 'h w c -> c h w')
    curr_images.append(img1)
    curr_images.append(img2)
    # curr_images.append(img3)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    
    return curr_image

 
def eval_bc(config, pt_name='model_0.pt', num_rollouts=20, video_output=None,
            video_fps=10, video_seconds=0, peg_pose=None):
    print("-------------------------")
    print(pt_name)
    print("-------------------------")
    set_seed(config['seed'])
    config['dataset_dir']=HOME_PATH+config['dataset_dir']
    model_interface = ModelInterface(config)
    model_interface.setup()
    # policy = IOUtils.load_policy_pt(config, HOME_PATH+'/ACT/ckpt_models/'+pt_name)
    policy = IOUtils.load_policy(config, pt_name)
    stats = IOUtils.load_stats(HOME_PATH+'/'+config['ckpt_dir'])
      

    run_episode(
        config,
        policy,
        stats,
        num_rollouts,
        video_output=video_output,
        video_fps=video_fps,
        video_seconds=video_seconds,
        peg_pose=peg_pose,
    )
    


def merge_act_numpy(actions_for_curr_step, k=0.01):
    """
    actions_for_curr_step: numpy array with shape (N, action_dim)
    返回加权求和的 raw action (1D numpy array length action_dim)
    """
    if actions_for_curr_step.size == 0:
        # 没有有效动作，返回全0
        return np.zeros((actions_for_curr_step.shape[1],), dtype=np.float32)
    # 过滤全0行
    populated = np.any(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[populated]
    if actions_for_curr_step.shape[0] == 0:
        return np.zeros((actions_for_curr_step.shape[1],), dtype=np.float32)
    n = actions_for_curr_step.shape[0]
    exp_weights = np.exp(-k * np.arange(n))
    exp_weights = exp_weights / exp_weights.sum()
    # 广播乘并求和
    raw_action = (actions_for_curr_step * exp_weights.reshape(-1, 1)).sum(axis=0)
    return raw_action


def run_episode(config, policy, stats, num_rollouts, video_output=None,
                video_fps=10, video_seconds=0, peg_pose=None):

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    chunk_size = config['chunk_size']
    Freq = 80  # hz
    mujoco_env = make_sim_env(freq=Freq)
    if peg_pose is None:
        peg_pose = sample_transfer_pose()  # 随机
    else:
        peg_pose = np.asarray(peg_pose, dtype=np.float64)
        if peg_pose.shape != (6,):
            raise ValueError(f"peg_pose must contain six values, got {peg_pose!r}")
        print(f"using fixed peg_pose: {peg_pose}")
    env = H1DexEnvInference(
        mujoco_env=mujoco_env,
        peg_pose=peg_pose
    )

    desired_dt = 1.0 / Freq  # 每步目标时间间隔（秒）
    action_dim = config['action_dim']  # 与模型输出维度对应
    device = next(policy.parameters()).device
    video_writer = None
    video_frames_written = 0
    max_video_frames = video_fps * video_seconds if video_seconds > 0 else None
    if video_output:
        output_path = Path(video_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    try:
        for rollout_id in range(num_rollouts):
            env.reset()

            max_times = 600

            if config['temporal_agg']:
            # all_time_actions: shape (T, T+chunk_size, action_dim)
                all_time_actions = np.zeros([max_times, max_times + chunk_size, action_dim], dtype=np.float32)
                query_frequency = 1
            else:
            # 不做 temporal agg 时，将按 chunk_size 执行（模型输出 chunk_size 个动作逐条执行）
                query_frequency = chunk_size

            print(f"{rollout_id+1}/{num_rollouts}")

        # 记录上一次推理得到的 output（当不使用 temporal_agg 时按索引执行）
            output_buffer = None
            output_index = 0
            last_camera_frame = None

            with torch.inference_mode():
                for t in tqdm(range(max_times)):
            # while t < max_times:
                    loop_start = time.perf_counter()

                    need_query = (t % query_frequency == 0)

                    if need_query:
                    # 采集观测
                        obs = np.zeros(32, dtype=np.float32)
                        obs[0:17] = env.get_state() * R2D
                        obs[17:32] = 0
                        qpos_numpy = obs.copy()
                        qpos = pre_process(qpos_numpy)
                        qpos_t = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

                        camera_frames = env.get_image(env.mujoco_env._get_image_obs())
                        camera_names = config['camera_names']
                        if len(camera_names) != 2 or any(name not in camera_frames for name in camera_names):
                            raise ValueError(
                                "H1 evaluation requires exactly two available cameras; "
                                f"got {camera_names!r}, available={list(camera_frames)!r}"
                            )
                        curr_image = get_image(
                            camera_frames[camera_names[0]],
                            camera_frames[camera_names[1]],
                            device,
                        )
                        # Keep a fixed, human-readable camera order in the MP4;
                        # this does not affect the policy tensor above.
                        last_camera_frame = np.hstack((camera_frames["top"], camera_frames["angle"]))

                        if config['policy_class'] in {"ACT", "ACTTV"}:


                            all_actions = policy(qpos_t, curr_image)  # tensor
                            if isinstance(all_actions, tuple) or isinstance(all_actions, list):
                                all_actions = all_actions[0]
                            all_actions_np = all_actions.detach().cpu().numpy()[0]  # (chunk_size, action_dim)

                            if config['temporal_agg']:
                            # 把 chunk 写入全局 buffer
                                start_idx = t
                                end_idx = t + chunk_size
                            # 防止越界
                                if end_idx > all_time_actions.shape[1]:
                                    end_idx = all_time_actions.shape[1]
                                    write_len = end_idx - start_idx
                                    if write_len > 0:
                                        all_time_actions[start_idx, start_idx:end_idx, :write_len and None] = all_actions_np[:write_len]
                                else:
                                    all_time_actions[start_idx, start_idx:end_idx, :] = all_actions_np
                            # 合并当前时刻的候选动作
                                actions_for_curr_step = all_time_actions[:, t, :]  # shape (T, action_dim)
                                raw_action = merge_act_numpy(actions_for_curr_step, k=0.1)
                            else:
                            # 不做 temporal agg：按 chunk 内顺序执行
                                output_buffer = all_actions_np  # shape (chunk_size, action_dim)
                                output_index = 0
                                raw_action = output_buffer[output_index]
                                output_index += 1
                        else:
                            raise NotImplementedError(f"Unknown policy_class {config['policy_class']}")
                    else:
                        if not config['temporal_agg'] and output_buffer is not None:
                            if output_index < output_buffer.shape[0]:
                                raw_action = output_buffer[output_index]
                                output_index += 1
                            else:
                                raw_action = output_buffer[-1]
                        else:
                            if config['temporal_agg']:
                                actions_for_curr_step = all_time_actions[:, t, :]
                                raw_action = merge_act_numpy(actions_for_curr_step, k=0.1)
                            else:
                                raw_action = np.zeros((action_dim,), dtype=np.float32)

                    # post process
                    action = post_process(raw_action)
                    env.step(action[:17])

                    if output_path is not None and last_camera_frame is not None:
                        if max_video_frames is None or video_frames_written < max_video_frames:
                            if video_writer is None:
                                height, width = last_camera_frame.shape[:2]
                                video_writer = cv2.VideoWriter(
                                    str(output_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    video_fps,
                                    (width, height),
                                )
                                if not video_writer.isOpened():
                                    raise RuntimeError(f"Cannot open video writer: {output_path}")
                            video_writer.write(last_camera_frame)
                            video_frames_written += 1

                    # 时间控制：确保每步近似 desired_dt
                    loop_end = time.perf_counter()
                    elapsed = loop_end - loop_start
                    to_sleep = desired_dt - elapsed
                    if to_sleep > 0:
                        time.sleep(to_sleep)
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Saved camera video: {output_path} ({video_frames_written} frames @ {video_fps} fps)")

         


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Train BC model with specified epoch number.')
    parser.add_argument('--epoch', type=int, required=True, 
                       help='Epoch number of the model checkpoint to load (e.g., 8000)')
    parser.add_argument('--num-rollouts', type=int, default=1,
                       help='Number of 600-step MuJoCo rollouts to execute (default: 1)')
    parser.add_argument('--config', type=str, help='Optional path to a YAML configuration file.')
    parser.add_argument('--video-output', type=str,
                       help='Optional MP4 path; records the combined top/angle MuJoCo cameras.')
    parser.add_argument('--video-fps', type=int, default=10,
                       help='Frame rate for --video-output (default: 10).')
    parser.add_argument('--video-seconds', type=int, default=0,
                       help='Maximum output-video duration; 0 records the full rollout.')
    parser.add_argument('--force-exit', action='store_true',
                       help='Exit with os._exit(0) after artifacts are flushed; use only '
                            'for an environment with a known native-library teardown crash.')
    parser.add_argument('--peg-pose', type=float, nargs=6, metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                       help='Optional fixed red-stick pose for reproducible evaluation; omit for random placement.')
    args = parser.parse_args()
    io_utils = IOUtils()
    config = io_utils.load_config(args.config)
    pt_name_ = "policy_epoch_" + str(args.epoch) + ".ckpt"
    print(pt_name_)
    if args.video_fps < 1 or args.video_seconds < 0:
        raise SystemExit('--video-fps must be positive and --video-seconds cannot be negative')
    eval_bc(
        config,
        pt_name=pt_name_,
        num_rollouts=args.num_rollouts,
        video_output=args.video_output,
        video_fps=args.video_fps,
        video_seconds=args.video_seconds,
        peg_pose=args.peg_pose,
    )
    if args.force_exit:
        # The MuJoCo/PyKDL combination in the current VM can crash only while
        # Python tears down native objects, after the rollout and MP4 writer
        # have completed.  Make this an explicit opt-in rather than masking
        # normal runtime failures.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()

    


