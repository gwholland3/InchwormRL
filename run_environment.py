import gymnasium as gym
import numpy as np

from agent import REINFORCE
from inchworm import InchwormEnv


def run_simulation_with_agent():
    env = InchwormEnv(render_mode="human")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(5e3)

    # Observation-space of Inchworm env
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of Inchworm env
    action_space_dims = env.action_space.shape[0]

    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
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

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

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


if __name__ == "__main__":
    run_simulation_with_agent()
