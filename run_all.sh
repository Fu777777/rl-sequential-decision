#!/bin/bash
# 全部实验：2 个环境 x 3 个算法 x 3 个种子，并行 8 个进程
P=${PYTHON:-python}
for env in CartPole-v1:100000 LunarLander-v3:200000; do
  for algo in dqn a2c ppo; do for seed in 0 1 2; do
    echo "$P run.py --env ${env%%:*} --steps ${env##*:} --algo $algo --seed $seed"
  done; done
done | xargs -P 8 -I{} sh -c "{}"
# 补充实验：A2C/PPO 采样便宜，给 LunarLander 1M 步预算（与 DQN 200k 的墙钟时间相近）
for algo in a2c ppo; do for seed in 0 1 2; do
  echo "$P run.py --env LunarLander-v3 --steps 1000000 --algo $algo --seed $seed --tag=-1M"
done; done | xargs -P 6 -I{} sh -c "{}"
