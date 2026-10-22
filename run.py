"""运行一组实验：python run.py --env CartPole-v1 --algo ppo --seed 0 --steps 100000
结果写入 results/<env>_<algo>_<seed>.json（回合结束步数与回报）。"""
import argparse
import json
import os
import time

import torch

from rlseq import actor_critic, dqn
from rlseq.common import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--algo", choices=["dqn", "a2c", "ppo"], default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--tag", default="", help="附加到算法名后，如 -1M，用于区分不同步数预算的补充实验")
    args = ap.parse_args()
    set_seed(args.seed); torch.set_num_threads(1)
    t0 = time.time()
    if args.algo == "dqn":
        curve = dqn.train(args.env, args.steps, args.seed, {})
    else:
        curve = actor_critic.train(args.env, args.steps, args.seed, {}, algo=args.algo)
    os.makedirs("results", exist_ok=True)
    out = f"results/{args.env}_{args.algo}{args.tag}_{args.seed}.json"
    json.dump({"curve": curve, "seconds": time.time() - t0}, open(out, "w"))
    last = [r for _, r in curve[-20:]]
    print(f"{args.env} {args.algo} seed{args.seed}: {len(curve)} 回合, 最后20回合均值 {sum(last) / len(last):.1f}, "
          f"用时 {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
