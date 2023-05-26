import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import torch

from agent import REINFORCE
from inchworm import InchwormEnv


def run_simulation_with_agent(render=False):
    env = InchwormEnv(render_mode=("human" if render else "rgb_array"))
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(
        env, 50
    )  # Records episode-reward

    total_num_episodes = int(5e3)

    # Observation-space of Inchworm env
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of Inchworm env
    action_space_dims = env.action_space.shape[0]

    agent = REINFORCE(obs_space_dims, action_space_dims)
    weights_file = "policy_network_weights.pth"
    weights_dir = os.path.dirname(weights_file)

    # Check if weights file exists
    if os.path.exists(weights_dir):
        # Load previous weights
        agent.net.load_state_dict(torch.load(weights_file))
        print("Previous weights loaded.")
    reward_over_episodes = []

    pbar = tqdm(range(total_num_episodes), unit="eps")

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
    try:
        os.makedirs(
            weights_dir, exist_ok=True
        )  # Create the directory if it doesn't exist
        weights_file = os.path.join(weights_dir, weights_file)
        torch.save(agent.net.state_dict(), weights_file)
        print("Weights saved successfully.")
    except Exception as e:
        print(f"Error occurred while saving weights: {e}")


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


if __name__ == "__main__":
    run_simulation_with_agent(False)
