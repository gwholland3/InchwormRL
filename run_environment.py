import os
import sys
import time
import gymnasium as gym
import keyboard
import numpy as np

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from inchworm import InchwormEnv

from glfw import GLFWError


def train_with_sb3_agent(
    model_name="inchworm_sac",
    algorithm: BaseAlgorithm = SAC,
    total_timesteps=30000,
    render=False,
):
    model_path = f"test_models/{model_name}.zip"
    env = InchwormEnv(render_mode=("human" if render else "rgb_array"))
    check_env(
        env
    )  # Make sure our env is compatible with the interface that stable-baselines3 agents expect

    try:
        model = algorithm.load(model_path, env)
        print("Continuing training of saved model")
    except FileNotFoundError:
        print("No saved model found, training new model")
        model = algorithm("MlpPolicy", env, verbose=1)

    model.set_random_seed(time.time_ns() % 2 ** 32)  # Set random seed to current time

    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("Interrupted by user, saving model")
    finally:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

    if render:
        input("Done. Press enter to view trained agent")

        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")

    else:
        print("Done.")

    env.close()


def run_simulation_with_sb3_agent(
    model_name="inchworm_sac", model_dir="saved_models", algorithm: BaseAlgorithm = SAC
):
    saved_model_path = f"{model_dir}/{model_name}.zip"
    env = InchwormEnv(render_mode="human")
    check_env(
        env
    )  # Make sure our env is compatible with the interface that stable-baselines3 agents expect

    try:
        model = algorithm.load(saved_model_path, env)
        print("Using specified model")
    except FileNotFoundError:
        print("Specified model not found")
        sys.exit(1)

    model.set_random_seed(time.time_ns() % 2 ** 32)  # Set random seed to current time

    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        try:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")
        except KeyboardInterrupt:
            break

    env.close()


def run_simulation_random():
    env = InchwormEnv(render_mode="human")

    # Must reset the env before making the first call to step()
    observation, info = env.reset()

    for _ in range(1000):
        # Select a random action from the sample space
        action = env.action_space.sample()

        # Apply that action to the environment, store the resulting data
        observation, reward, terminated, truncated, info = env.step(action)

        # End current iteration if necessary
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def run_simulation_control():
    env = InchwormEnv(render_mode="human")

    # Must reset the env before making the first call to step()
    observation, info = env.reset()

    while True:
        # Determine action
        action = get_action()

        # Break on 'q' press
        if action is None:
            break

        # Apply that action to the environment
        env.step(action)
    env.close()


def get_action():
    if keyboard.is_pressed("q"):
        return None

    action = []
    if keyboard.is_pressed("j"):
        action.append(1)
    elif keyboard.is_pressed("u"):
        action.append(-1)
    else:
        action.append(0)

    if keyboard.is_pressed("i"):
        action.append(1)
    elif keyboard.is_pressed("k"):
        action.append(-1)
    else:
        action.append(0)

    if keyboard.is_pressed("l"):
        action.append(1)
    elif keyboard.is_pressed("o"):
        action.append(-1)
    else:
        action.append(0)

    action.append(1 if keyboard.is_pressed("[") else -1)
    action.append(1 if keyboard.is_pressed("]") else -1)

    return np.array(action)


if __name__ == "__main__":
    # run_simulation_random()

    # train_with_sb3_agent(          # train a new model with SAC
    #     model_name="inchworm_sac",
    #     algorithm=SAC,
    #     total_timesteps=2000000
    # )

    # train_with_sb3_agent(          # train a new model with TD3
    #     model_name="inchworm_td3",
    #     algorithm=TD3,
    #     total_timesteps=10000000
    # )

    # run_simulation_with_sb3_agent(   # run a local TD3 test model
    #     model_name="inchworm_td3",
    #     model_dir="test_models",
    #     algorithm=TD3
    # )

    run_simulation_with_sb3_agent(   # run a TD3 saved model
        model_name="inchworm1.1_td3",
        algorithm=TD3
    )

    # run_simulation_with_sb3_agent(model_name="inchworm_sac", model_dir="test_models")  # run a local test model
    # run_simulation_with_sb3_agent(model_name="naive_1mtts")                            # run a saved model

    # run_simulation_control()
