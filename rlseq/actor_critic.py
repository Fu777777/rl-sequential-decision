"""A2C 与 PPO（同步多环境、GAE、共享代码）。

- A2C：每次 rollout(n_steps 很短) 只做 1 次梯度更新，损失 = -logπ·A + c_v·(V-G)^2 - c_e·H
- PPO：rollout 较长，同一批数据做多轮小批量更新，用裁剪比率 clip(r,1±ε) 限制策略变化
截断的回合用 V(final_obs) 自举，终止的回合不自举。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .common import EpisodeLogger, make_vec, mlp


class ActorCritic(nn.Module):
    def __init__(self, od, na):
        super().__init__()
        self.pi, self.v = mlp(od, na), mlp(od, 1)

    def dist_value(self, x):
        return Categorical(logits=self.pi(x)), self.v(x).squeeze(-1)


def train(env_id, total_steps, seed, cfg, algo="ppo"):
    ppo = algo == "ppo"
    n_envs = cfg.get("n_envs", 8)
    n_steps = cfg.get("n_steps", 128 if ppo else 8)
    gamma, lam = cfg.get("gamma", 0.99), cfg.get("lam", 0.95 if ppo else 1.0)
    envs = make_vec(env_id, n_envs, seed)
    od, na = envs.single_observation_space.shape[0], envs.single_action_space.n
    net = ActorCritic(od, na)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 3e-4 if ppo else 7e-4))
    log = EpisodeLogger(n_envs)
    obs, _ = envs.reset(seed=seed)
    step = 0
    while step < total_steps:
        # ---------- 采样 rollout ----------
        B = {k: [] for k in ("s", "a", "logp", "v", "r", "d")}
        for _ in range(n_steps):
            s = torch.from_numpy(obs.astype(np.float32))
            with torch.no_grad():
                dist, v = net.dist_value(s)
                a = dist.sample()
            obs, r, term, trunc, info = envs.step(a.numpy())
            log.step(r, term | trunc, step + n_envs)  # 用原始奖励记录回合回报
            r = r.astype(np.float32)
            if trunc.any():  # 截断：把 γV(末状态) 加进奖励，相当于自举
                fo = torch.from_numpy(np.stack(info["final_obs"][trunc]).astype(np.float32))
                with torch.no_grad():
                    r[trunc] += gamma * net.dist_value(fo)[1].numpy() * (~term[trunc])
            step += n_envs
            for k, x in zip(B, (s, a, dist.log_prob(a), v, torch.from_numpy(r), torch.from_numpy((term | trunc).astype(np.float32)))):
                B[k].append(x)
        B = {k: torch.stack(x) for k, x in B.items()}  # [T, n_envs]
        with torch.no_grad():
            next_v = net.dist_value(torch.from_numpy(obs.astype(np.float32)))[1]
        # ---------- GAE(λ) ----------
        adv, gae = torch.zeros_like(B["r"]), 0
        for t in reversed(range(n_steps)):
            nv = next_v if t == n_steps - 1 else B["v"][t + 1]
            delta = B["r"][t] + gamma * nv * (1 - B["d"][t]) - B["v"][t]
            gae = delta + gamma * lam * (1 - B["d"][t]) * gae
            adv[t] = gae
        ret = adv + B["v"]
        s, a, lp_old, adv, ret = (x.flatten(0, 1) for x in (B["s"], B["a"], B["logp"], adv, ret))
        # ---------- 更新 ----------
        epochs, mb = (cfg.get("epochs", 10), 256) if ppo else (1, len(s))
        for _ in range(epochs):
            for idx in torch.randperm(len(s)).split(mb):
                dist, v = net.dist_value(s[idx])
                lp, A = dist.log_prob(a[idx]), adv[idx]
                if ppo:
                    A = (A - A.mean()) / (A.std() + 1e-8)
                    ratio = (lp - lp_old[idx]).exp()
                    pl = -torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                else:
                    pl = -(lp * A).mean()
                loss = pl + 0.5 * (v - ret[idx]).pow(2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    return log.curve
