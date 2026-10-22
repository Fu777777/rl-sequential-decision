"""DQN：经验回放 + 目标网络（软更新）+ epsilon 线性衰减 + Huber 损失。截断(time limit)不当作终止。"""
import numpy as np
import torch
import torch.nn as nn

from .common import EpisodeLogger, make_vec, mlp


def train(env_id, total_steps, seed, cfg):
    envs = make_vec(env_id, 1, seed)
    od, na = envs.single_observation_space.shape[0], envs.single_action_space.n
    q, tq = mlp(od, na, 256), mlp(od, na, 256)
    tq.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.get("lr", 5e-4))
    N = cfg.get("buffer", 100_000)
    S, S2 = np.zeros((N, od), np.float32), np.zeros((N, od), np.float32)
    A, R, D = np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, np.float32)
    gamma, batch, ptr, size = cfg.get("gamma", 0.99), 128, 0, 0
    eps_steps = int(total_steps * cfg.get("eps_frac", 0.3))
    log, rng = EpisodeLogger(1), np.random.default_rng(seed)
    obs, _ = envs.reset(seed=seed)
    for t in range(total_steps):
        eps = max(0.05, 1.0 - 0.95 * t / eps_steps)
        if rng.random() < eps:
            a = envs.action_space.sample()
        else:
            with torch.no_grad():
                a = q(torch.from_numpy(obs.astype(np.float32))).argmax(1).numpy()
        obs2, r, term, trunc, info = envs.step(a)
        real_next = info["final_obs"][0] if (term[0] or trunc[0]) else obs2[0]
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = obs[0], a[0], r[0], real_next, float(term[0])
        ptr, size = (ptr + 1) % N, min(size + 1, N)
        log.step(r, term | trunc, t + 1)
        obs = obs2
        if size >= 1000:
            idx = rng.integers(0, size, batch)
            s, a_, r_, s2, d = map(torch.from_numpy, (S[idx], A[idx], R[idx], S2[idx], D[idx]))
            with torch.no_grad():
                y = r_ + gamma * (1 - d) * tq(s2).max(1).values
            loss = nn.functional.smooth_l1_loss(q(s).gather(1, a_[:, None]).squeeze(1), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
            with torch.no_grad():  # 软更新目标网络 tau=0.005
                for p, tp in zip(q.parameters(), tq.parameters()):
                    tp.mul_(0.995).add_(0.005 * p)
    return log.curve
