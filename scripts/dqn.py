
"""
from miniworld.dqn import dqn_solver
import argparse
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="LunarLander-v2")
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"
    env = gym.make(args.env_name, view=view_mode, render_mode="human")

    dqn = dqn_solver(env)
    dqn.run()

"""
"""


import gymnasium as gym
from stable_baselines3 import DQN
from  miniworld.envs.maze import MazeS3
import pyglet
import time

# Create and wrap the environment
#LunarLander-v2
#MiniWorld-MazeS3-v0

model = DQN("CnnPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=100000)

# Enjoy trained agent
obs = env.reset()
env.render()
for i in range(1000):
    time.sleep(1)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from  miniworld.envs.maze import MazeS3
import argparse
import miniworld
from miniworld.dqn_test import DQN_solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-Hallway-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    dqn_control = DQN_solver(env, args.no_time_limit, args.domain_rand)
    dqn_control.run()

if __name__ == "__main__":
    main()
