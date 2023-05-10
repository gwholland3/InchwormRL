import gymnasium as gym
import mujoco

def demo_program():
    env = gym.make("Ant-v4", render_mode="human")

    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def inchworm_program():
    model = mujoco.MjModel.from_xml_path("inchworm.xml")
    data = mujoco.MjData(model)
    while data.time < 1:
        mujoco.mj_step(model, data)
        print(data.geom_xpos)


if __name__ == "__main__":
    inchworm_program()
