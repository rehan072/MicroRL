#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3.dqn import DQN
import optuna

from stable_baselines3.common.callbacks import EvalCallback


import math
import pyglet
from pyglet.window import key
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


from miniworld.envs.maze import Maze, MazeS2, MazeS3, MazeS3Fast

from typing import Callable
import pyglet
import time
import numpy as np
import torch
"""
#env = gym.make("MiniWorld-MazeS3-v0", render_mode="human")
#model = DQN("CnnPolicy", env, verbose=1)


class dqn_solver():
    def __init__(self, env, policy="MlpPolicy"):
        self.env = env
        self.model = DQN(policy, self.env, verbose=1)
        self.model.learn(total_timesteps=int(2e5), progress_bar=True)
        
        #self.vec_env = self.model.get_env()
        #elf.obs = self.vec_env.reset()

    def run(self):
        observation, info = self.env.reset(seed=42)
        self.vec_env = self.model.get_env().reset()
        for _ in range(1000):
            action, _states = self.model.predict(observation, deterministic=True)
            observation, rewards, done, info = self.vec_env.step(action) 
            self.env.render(mode="human")
        print("here")
        #pyglet.app.run()
        self.vev_env.close()


"""

##change
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class DQN_solver:
    def __init__(self, env, no_time_limit, domain_rand):
        self.env = env
        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

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

    def run(self):


        def model_train(trial):
            #check_env(self.env, warn=True)

            model_params = self.optimize_dqn2(trial)
            #evalcal = EvalCallback(self.env, n_eval_episodes=1, eval_freq= 150, render=True)
            model = DQN("CnnPolicy", self.env, verbose = 1)
            # Train the agent
            #model.learn(total_timesteps=int(1e5), progress_bar=True, callback=evalcal)
            model.learn(total_timesteps=int(1e5))
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        
            return -1 * mean_reward

        obs, info = self.env.reset()

        env = self.env
        # Create the display window
        self.env.render()

        #begin of insert
        check_env(self.env,warn=True)
        study = optuna.create_study()
        try:
            print("here")
            study.optimize(model_train, n_trials=100, n_jobs=1)
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')

        model = DQN.load("mouse",env=self.env)
        #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        
        print("learnt")
        # Enjoy trained agent
        obs, info = self.env.reset()
        for _ in range(500):
            time.sleep(0.1)
            action, _states = model.predict(obs)
            print(action)
            #obs, rewards, dones, info = 
            obs = self.step(action)
            #env.render()
        #end of insert

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.UP:
                self.step(self.env.actions.move_forward)
            elif symbol == key.DOWN:
                self.step(self.env.actions.move_back)
            elif symbol == key.LEFT:
                self.step(self.env.actions.turn_left)
            elif symbol == key.RIGHT:
                self.step(self.env.actions.turn_right)
            elif symbol == key.PAGEUP or symbol == key.P:
                self.step(self.env.actions.pickup)
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.step(self.env.actions.drop)
            elif symbol == key.ENTER:
                self.step(self.env.actions.done)

            """
            pass
            

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        print(
            "step {}/{}: {}".format(
                self.env.step_count + 1,
                self.env.max_episode_steps,
                self.env.actions(action).name,
            )
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()

        self.env.render()
        return obs
