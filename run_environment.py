import time
import gymnasium as gym
from mujoco import MjModel, MjData, viewer, mj_step


def demo_program():
    env = gym.make("Ant-v4", render_mode="human", xml_file='<INSERT ABSOLUTE PATH TO INCHWORM.XML FILE HERE>')

    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def inchworm_program():
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
    demo_program()
