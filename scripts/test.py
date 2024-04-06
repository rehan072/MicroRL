import gymnasium as gym
from stable_baselines3.dqn import DQN
import optuna

from stable_baselines3.common.callbacks import EvalCallback

import math
import pyglet
from pyglet.window import key
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


from miniworld.envs.maze import Maze, MazeS2, MazeS3, MazeS3Fast
from miniworld.envs.hallway import Hallway
from typing import Callable
import pyglet
import time
import numpy as np
import torch

env = gym.make("MiniWorld-Hallway-v0")

def optimize_dqn2(trial):
        """ Learning hyperparamters we want to optimise"""
        return {
            'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.00001, 0.01),
            'gamma': trial.suggest_loguniform('gamma',0.1, 1),
            'exploration_fraction': trial.suggest_uniform('exploration_fraction', 0.01, 0.5),
            'exploration_initial_eps': trial.suggest_uniform('exploration_initial_eps', 0.5, 1.0),
            'exploration_final_eps': trial.suggest_uniform('exploration_final_eps', 0.01, 0.5)
        }

def optimize_dqn3(trial):
        """ Learning hyperparamters we want to optimise"""
        return {
            
            'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.01),
            'gamma': trial.suggest_float('gamma',0.1, 1),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.01, 0.5),
            'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.5, 1.0),
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.5)
        }
def model_train(trial):
            #check_env(self.env, warn=True)
            
            model_params = optimize_dqn3(trial)
            #evalcal = EvalCallback(self.env, n_eval_episodes=1, eval_freq= 150, render=True)
            model = DQN("CnnPolicy", env, verbose = 1)
            # Train the agent
            #model.learn(total_timesteps=int(1e5), progress_bar=True, callback=evalcal)
            model.learn(total_timesteps=int(1e5))
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        
            return -1 * mean_reward

if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(model_train, n_trials=100, n_jobs=4)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')