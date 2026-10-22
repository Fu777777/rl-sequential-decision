"""公共工具：MLP、随机种子、向量化环境、回合回报记录器。"""
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_vec(env_id, n, seed):
    """SAME_STEP 自动重置：回合结束的那一步就返回新回合的 obs，真正的末状态放在 info['final_obs']。"""
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    return envs


class EpisodeLogger:
    """累计每个并行环境的回合回报，回合结束时记录 (全局步数, 回报)。"""

    def __init__(self, n):
        self.ret, self.curve = np.zeros(n), []

    def step(self, rewards, dones, global_step):
        self.ret += rewards
        for i in np.flatnonzero(dones):
            self.curve.append((global_step, float(self.ret[i])))
            self.ret[i] = 0.0
