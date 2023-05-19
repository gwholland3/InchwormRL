import time
import gymnasium as gym
from mujoco import MjModel, MjData, viewer, mj_step
from inchworm import InchwormEnv


def run_inchworm_simulation():
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


def raw_mujoco_program():
    model = MjModel.from_xml_path("inchworm.xml")
    data = MjData(model)
    with viewer.launch_passive(model, data) as view:
        start = time.time()
        while view.is_running() and time.time() - start < 30:
            step_start = time.time()
            mj_step(model, data)

            with view.lock():
                # modify viewer options here
                pass
            view.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_inchworm_simulation()
