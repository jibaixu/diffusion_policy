"""Validation of maniskill for different tasks

"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional, Dict
import gymnasium as gym
import numpy as np
import torch
import tyro
import math
import json
import os
import hydra
import dill
from transformers import AutoTokenizer, AutoModel
from collections import deque

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

ROBOT = "panda" # ["panda", "widowxai", "xarm6", "xarm7"]
EPOCH = 20

BENCHMARK_ENVS = ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1",]
ENV_INSTRUCTION_MAP = {
    "PickCube-v1": "Pick up the cube.",
    "PushCube-v1": "Push the cube to the target position.",
    "StackCube-v1": "Stack the cube on top of the other cube.",
    "PullCube-v1": "Push the cube to the target position.",
    "PullCubeTool-v1": "Pick up the cube tool and use it to bring the cube closer.",
    "PlaceSphere-v1": "Pick up the ball and place it in the target position.",
    "LiftPegUpright-v1": "Pick up the peg and place it upright.",
}
ENV_MAXSTEP_MAP = {
    "PickCube-v1": 500,
    "PushCube-v1": 500,
    "StackCube-v1": 500,
    "PullCube-v1": 500,
    "PullCubeTool-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 700,
}


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclass
class EvalConfig:
    """Configuration for evaluation

    """
    """Policy Path"""
    pretrain_policy_path = f"outputs/train/AllTasks-v3/{ROBOT}/checkpoints/epoch_{EPOCH}.ckpt"
    """Inference Device"""
    model_device = "cuda:0"
    
    resize_size: int = 224
    replan_steps: int = 5
    # env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = BENCHMARK_ENVS[INDEX]
    """Environment ID"""
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: Optional[int] = 60
    sim_freq: Optional[int] = 120
    num_cams: Optional[int] = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: Optional[int] = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: Optional[int] = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = True
    """Whether to save videos"""
    save_results: Optional[str] = None
    """Path to save results to. Should be path/to/results.csv"""
    save_path: str = f"outputs/eval/AllTasks-v3/{ROBOT}/epoch_{EPOCH}"
    shader: str = "default"
    num_per_task: int = 50


class MultiStepObservationQueue:
    def __init__(self, n_obs_steps: int = 2):
        self.n_obs_steps = n_obs_steps
        self.image_queue = deque([], maxlen=n_obs_steps)
        self.wrist_image_queue = deque([], maxlen=n_obs_steps)
        self.state_queue = deque([], maxlen=n_obs_steps)

    def __len__(self):
        return len(self.image_queue)

    def reset(self):
        self.image_queue.clear()
        self.wrist_image_queue.clear()
        self.state_queue.clear()

    def push(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray):
        if len(self.image_queue) != self.n_obs_steps:
            #! 队列为空时拷贝填充
            while len(self.image_queue) < self.n_obs_steps:
                self.image_queue.append(image)
                self.wrist_image_queue.append(wrist_image)
                self.state_queue.append(state)
        else:
            self.image_queue.append(image)
            self.wrist_image_queue.append(wrist_image)
            self.state_queue.append(state)

    def get_observation(self):
        """Concatenate all observations from the queue
        
        Returns: {
            "image":        ndarray[B=1, To=2, c, h, w], device=cpu
            "wrist_image":  ndarray[B=1, To=2, c, h, w], device=cpu
            "state":        ndarray[B=1, To=2, D=8], device=cpu
        }
        """
        return {
            "image": np.concatenate(list(self.image_queue), axis=1),
            "wrist_image": np.concatenate(list(self.wrist_image_queue), axis=1),
            "state": np.concatenate(list(self.state_queue), axis=1),
        }


def main(args: EvalConfig):
    assert args.cpu_sim, "Must be CPU Sim"

    os.makedirs(args.save_path, exist_ok=True)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    device = torch.device(args.model_device)
    #!############### 构造文本编码器 ###############
    # langage model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    lang_model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float32)
    lang_model.to(device)
    def get_language_embedding(language):
        with torch.no_grad():
            encoded_lang = tokenizer(language,padding = True,truncation=True, return_tensors='pt').to(device)
            outputs = lang_model(**encoded_lang)
            language_embedding = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(1)      # (B, 1, 768)
        if language_embedding.shape[-1] != 768 and len(language_embedding.shape):
            language_embedding = language_embedding.reshape(-1,1,768)
        return language_embedding

    #!############### 构造模型 ###############
    # load checkpoint
    payload = torch.load(open(args.pretrain_policy_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=args.save_path)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    # tranfer device
    policy.to(device)
    policy.eval()

    #!############### 构造观测队列和动作队列 ###############
    obs_queue = MultiStepObservationQueue(n_obs_steps=cfg.n_obs_steps)
    action_queue = deque()

    kwargs = dict()
    # if args.env_id in BENCHMARK_ENVS:
    #     kwargs = dict(
    #         camera_width=args.cam_width,
    #         camera_height=args.cam_height,
    #         num_cameras=args.num_cams,
    #     )

    total_successes = 0.0
    success_dict = {}
    for env_id in BENCHMARK_ENVS:
        if args.cpu_sim:
            def make_env():
                def _init():
                    env = gym.make(env_id,
                                obs_mode=args.obs_mode,
                                sim_config=sim_config,
                                robot_uids="panda_wristcam",
                                sensor_configs=dict(shader_pack=args.shader),
                                human_render_camera_configs=dict(shader_pack=args.shader),
                                viewer_camera_configs=dict(shader_pack=args.shader),
                                render_mode=args.render_mode,
                                control_mode=args.control_mode,
                                **kwargs)
                    env = CPUGymWrapper(env, )
                    return env
                return _init
            # mac os system does not work with forkserver when using visual observations
            env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
            base_env = make_env()().unwrapped

        base_env.print_sim_details()
        
        task_successes = 0.0
        for seed in range(args.num_per_task):
            images = []
            video_nrows = int(np.sqrt(num_envs))
            with torch.inference_mode():
                #! 任务开始前重置观测和动作队列
                obs_queue.reset()
                action_queue.clear()

                env.reset(seed=seed+2025)
                env.step(env.action_space.sample())  # warmup step
                obs, info = env.reset(seed=seed+2025)
                if args.save_video:
                    images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                    # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())

                task_description = ENV_INSTRUCTION_MAP[env_id]
                step_length = ENV_MAXSTEP_MAP[env_id]

                #!############### 语言指令编码 ###############
                lang_emb = get_language_embedding(task_description) # tensor[B=1, 1, 768]
                lang_emb = torch.concatenate([lang_emb, lang_emb], dim=0).unsqueeze(0) # tensor[B=1, To=2, 1, 768]
                lang_emb = lang_emb.cpu().numpy()

                N = step_length
                with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
                    for i in range(N):
                        if args.cpu_sim:
                            #!############### 构造输入数据 ###############
                            #! element = {
                            #!    "obs": {
                            #!       "image":       tensor[B=1, To=2, c, h, w], deivce=cuda, float[0, 1]
                            #!       "wrist_image": tensor[B=1, To=2, c, h, w], device=cuda, float[0, 1]
                            #!       "state":       tensor[B=1, To=2, D=8], device=cuda
                            #!    },
                            #!    "lang_emb": tensor[B=1, To=2, 1, D=768], device=cuda
                            #! }
                            image = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                            wrist_image = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])
                            state = np.expand_dims(
                                        np.concatenate(
                                                (
                                                    obs["extra"]["tcp_pose"],
                                                    obs["agent"]["qpos"][-1:],
                                                )
                                            ),
                                        axis=0,
                                    )
                            
                            # (h, w, c) -> (c, h, w) and normalization
                            image = image.transpose(2, 0, 1).astype(np.float32) / 255
                            wrist_image = wrist_image.transpose(2, 0, 1).astype(np.float32) / 255

                            # (c, h, w) -> (B=1, To=1, c, h, w)
                            image = image[None, :][None, :]
                            wrist_image = wrist_image[None, :][None, :]
                            state = state[None, :]

                            obs_queue.push(
                                image=image,
                                wrist_image=wrist_image,
                                state=state,
                            )
                            obs_dict = obs_queue.get_observation()

                            element = {
                                "obs": obs_dict,
                                "lang_emb": lang_emb,
                            }
                            element = dict_apply(
                                element, lambda x: torch.from_numpy(x).to(device))

                        if len(action_queue) <= 0:
                            action_dict = policy.predict_action(element)
                            action_chunk = action_dict['action'].cpu().numpy()[0] # action_dict['action']: tensor[B=1, n_action_steps, D]
                            action_queue.extend(action_chunk)
                        
                        action = action_queue.popleft()
  
                        obs, rew, terminated, truncated, info = env.step(action)
                        if args.save_video:
                            images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                            # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                        terminated = terminated if args.cpu_sim else terminated.item()
                        if terminated:
                            task_successes += 1
                            total_successes += 1

                profiler.log_stats("env.step")

                if args.save_video:
                    images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                    images_to_video(
                        images,
                        output_dir=args.save_path,
                        video_name=f"{env_id}-{seed}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}--success={terminated}",
                        fps=30,
                    )
                    del images
        env.close()
        print(f"Task Success Rate: {task_successes / args.num_per_task}")
        success_dict[env_id] = task_successes / args.num_per_task
    print(f"Total Success Rate: {total_successes / (args.num_per_task * len(BENCHMARK_ENVS))}")
    success_dict['total_success'] = total_successes / (args.num_per_task * len(BENCHMARK_ENVS))
    with open(f"{args.save_path}/success_dict.json", "w") as f:
        json.dump(success_dict, f)
    

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
