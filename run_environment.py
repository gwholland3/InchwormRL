import os
import sys
import gymnasium as gym
import keyboard
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from agent import REINFORCE
from inchworm import InchwormEnv

def train_with_sb3_agent(model_name="inchworm_sac", total_timesteps=30000, render=False):
    model_path = f"test_models/{model_name}.zip"
    env = InchwormEnv(render_mode=("human" if render else "rgb_array"))
    check_env(env)  # Make sure our env is compatible with the interface that stable-baselines3 agents expect

    try:
        model = SAC.load(model_path, env)
        print("Continuing training of saved model")
    except FileNotFoundError:
        print("No saved model found, training new model")
        model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
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


def run_simulation_with_sb3_agent(model_name="inchworm_sac", model_dir="saved_models"):
    saved_model_path = f"{model_dir}/{model_name}.zip"
    env = InchwormEnv(render_mode="human")
    check_env(env)  # Make sure our env is compatible with the interface that stable-baselines3 agents expect
    
    try:
        model = SAC.load(saved_model_path, env)
        print("Using specified model")
    except FileNotFoundError:
        print("Specified model not found")
        sys.exit(1)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")

    env.close()

def run_simulation_with_custom_agent(render=False):
    env = InchwormEnv(render_mode=("human" if render else "rgb_array"))
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(5e3)

    # Observation-space of Inchworm env
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of Inchworm env
    action_space_dims = env.action_space.shape[0]

    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    pbar = tqdm(
        range(total_num_episodes),
        unit="eps"
    )

    for episode in pbar:
        # Must reset the env before making the first call to step()
        observation, info = wrapped_env.reset()

        done = False
        while not done:
            # Request an action from the agent
            action = agent.sample_action(observation)

            # Apply that action to the environment, store the resulting data
            observation, reward, terminated, truncated, info = wrapped_env.step(action)

            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        avg_reward = int(np.mean(wrapped_env.return_queue))
        pbar.set_description(f"ep {episode}, avgReward {avg_reward}")

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
    if keyboard.is_pressed('q'):
        return None

    action = []
    if keyboard.is_pressed('j'):
        action.append(1)
    elif keyboard.is_pressed('u'):
        action.append(-1)
    else:
        action.append(0)

    if keyboard.is_pressed('i'):
        action.append(1)
    elif keyboard.is_pressed('k'):
        action.append(-1)
    else:
        action.append(0)

    if keyboard.is_pressed('l'):
        action.append(1)
    elif keyboard.is_pressed('o'):
        action.append(-1)
    else:
        action.append(0)

    action.append(1)
    action.append(-1)

    return np.array(action)


if __name__ == "__main__":
    # run_simulation_random()
    # run_simulation_with_custom_agent(True)

    # train_with_sb3_agent(model_name="inchworm_sac", total_timesteps=1000000)           # train a new model
    # run_simulation_with_sb3_agent(model_name="inchworm_sac", model_dir="test_models")  # run a local test model
    # run_simulation_with_sb3_agent(model_name="naive_1mtts")                            # run a saved model

    run_simulation_control()
