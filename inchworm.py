from collections import deque
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 8.0,
}


class InchwormEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is for CSC 480.

    The inchworm is a 2D robot consisting of four links attached in a line, with
    rotational joints between each link. The goal is to coordinate the four links
    to move in the forward (right) direction by applying torques on the three
    hinges connecting the links together and controlling the adhesion actuators
    on the two feet.

    ## Action Space
    The action space is a `Box([-1, -1, -1, 0, 0], 1, (5,), float32)`.
    An action represents the torques applied at the three hinge joints concatenated
    with the input applied to the two adhesion actuators.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint    | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | -------- | ------------ |
    | 0   | Torque applied on the rotor between the first and second links    | -1          | 1           | left_joint                       | hinge    | torque (N m) |
    | 1   | Torque applied on the rotor between the second and third links    | -1          | 1           | middle_joint                     | hinge    | torque (N m) |
    | 2   | Torque applied on the rotor between the third and fourth links    | -1          | 1           | right_joint                      | hinge    | torque (N m) |
    | 3   | Whether adhesion is activated on the left foot                    | 0           | 1           | left_foot                        | adhesion | force (N) |
    | 4   | Whether adhesion is activated on the right foot                   | 0           | 1           | right_foot                       | adhesion | force (N) |

    ## Observation Space

    Observations consist of positional values of different body parts of the inchworm,
    followed by the velocities of those individual parts (their derivatives) with all
    the positions ordered before all the velocities.

    By default, an observation is a `ndarray` with shape `(12,)`
    where the elements correspond to the following:

    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | y-orientation of the left foot                               | -Inf   | Inf    | hc_joint                               | hinge | angle (deg)              |
    | 1   | x-coordinate of the left foot                                | -Inf   | Inf    | hsx_joint                              | slide | position (m)             |
    | 2   | z-coordinate of the left foot                                | -Inf   | Inf    | hsz_joint                              | slide | position (m)             |
    | 3   | angle between the first and second segments                  | -Inf   | Inf    | left_joint                             | hinge | angle (deg)              |
    | 4   | angle between the second and third segments                  | -Inf   | Inf    | middle_joint                           | hinge | angle (deg)              |
    | 5   | angle between the third and fourth segments                  | -Inf   | Inf    | right_joint                            | hinge | angle (deg)              |
    | 6   | y-coordinate angular velocity of the left foot               | -Inf   | Inf    | hc_joint                               | hinge | angular velocity (deg/s) |
    | 7   | x-coordinate velocity of the left foot                       | -Inf   | Inf    | hsx_joint                              | slide | velocity (m/s)           |
    | 8   | z-coordinate velocity of the left foot                       | -Inf   | Inf    | hsz_joint                              | slide | velocity (m/s)           |
    | 9   | angular velocity of angle between first and second segments  | -Inf   | Inf    | left_joint                             | hinge | angular velocity (deg/s) |
    | 10  | angular velocity of angle between second and third segments  | -Inf   | Inf    | middle_joint                           | hinge | angular velocity (deg/s) |
    | 11  | angular velocity of angle between third and fourth segments  | -Inf   | Inf    | right_joint                            | hinge | angular velocity (deg/s) |

    The (x,y,z) coordinates are translational DOFs while the orientations are rotational
    DOFs expressed as quaternions. One can read more about free joints on the [Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    ## Rewards
    The reward consists of three parts:

    - *healthy_reward*: Every timestep that the inchworm is healthy (see definition in section "Episode Termination"), it gets a reward of fixed value `healthy_reward`

    - *forward_reward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the time
    between actions and is dependent on the `frame_skip` parameter (default is 5),
    where the frametime is 0.01 - making the default *dt = 5 * 0.01 = 0.05*.
    This reward would be positive if the inchworm moves forward (in positive x direction).

    - *ctrl_cost*: A negative reward for penalising the inchworm if it takes actions for motors
    that are too large. It is measured as *`ctrl_cost_weight` * sum(action[:3]<sup>2</sup>)*
    where *`ctr_cost_weight`* is a parameter set for the control and has a default value of 0.5.

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.

    `info` will also contain the individual reward terms.

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, ..., 0.0) with a uniform noise in the range
    of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional values and standard normal noise
    with mean 0 and standard deviation `reset_noise_scale` added to the velocity values for
    stochasticity. The initial orientation is designed to make the inchworm face forward.

    ## Episode End
    The inchworm is said to be unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite

    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode ends when any of the following happens:

    1. Truncation: The episode duration reaches 1000 timesteps
    2. Termination: The inchworm is unhealthy

    If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

    ## Arguments

    | Parameter                  | Type      | Default          | Description                   |
    |----------------------------|-----------|------------------|-------------------------------|
    | `xml_file`                 | **str**   | `"inchworm.xml"` | Path to a MuJoCo model |
    | `ctrl_cost_weight`         | **float** | `0.5`            | Weight for *ctrl_cost* term (see section on reward) |
    | `healthy_reward`           | **float** | `1`              | Constant reward given if the inchworm is "healthy" after timestep |
    | `terminate_when_unhealthy` | **bool**  | `True`           | If true, issue a done signal if the inchworm is deemed to be "unhealthy" |
    | `reset_noise_scale`        | **float** | `0.1`            | Scale of random perturbations of initial position and velocity (see section on Starting State) |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    root_body = "left_foot"

    from os import path

    inchworm_xml_file = path.join(path.dirname(__file__), "inchworm.xml")

    def __init__(
        self,
        xml_file=inchworm_xml_file,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        reset_noise_scale=0.1,
        velocity_record_length=100,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            reset_noise_scale,
            **kwargs,
        )

        # Store parameters
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._reset_noise_scale = reset_noise_scale

        obs_shape = 12

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        # How many frames to apply an action for when that action is applied to the environment
        frame_skip = 5

        # Store velocity history to calculate average velocity
        self.velocity_record = deque(maxlen=velocity_record_length)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _set_action_space(self):
        """
        Overriding this method so that we can manually set our action
        space to be in the range [-1, 1] for all actuators. This consistent
        and somewhat normalized action space helps the performance of many
        RL algorithms
        """
        num_actuators = self.model.nu
        self.action_space = Box(
            low=-1, high=1, shape=(num_actuators,), dtype=np.float32
        )
        return self.action_space

    @property
    def avg_velocity(self) -> float:
        return np.mean(self.velocity_record)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        """
        Control cost is a penalty for applying large forces to the hinge motors
        (the first three values of the action)
        """
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action[:3]))
        return control_cost

    @property
    def is_healthy(self):
        # State vector contains all the positions and velocities
        state = self.state_vector()
        is_healthy = np.isfinite(state).all()
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        """
        Correct the inputs to the adhesion actuators so that
        they are either 0 (off) or 1 (on). They are given in
        the range [-1, 1]
        """
        rescaled_adhesion_actions = (action[3:5] + 1) / 2  # Rescale to [0, 1]
        action[3:5] = np.round(rescaled_adhesion_actions)  # Round to 0 or 1

        # Record current robot position, apply the action to the simulation, then record the resulting robot position
        x_position_before = self.get_body_com(self.root_body)[0].copy()
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.get_body_com(self.root_body)[0].copy()

        # Calculate the robot's forward (x-axis) velocity based on its change in position
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Record the robot's velocity
        # self.velocity_record.append(x_velocity)

        # Calculate positive rewards
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        # Calculate penalties
        ctrl_cost = self.control_cost(action)

        costs = ctrl_cost

        # Calculate total reward to give to the agent
        reward = rewards - costs
        self.displacement = max(self.displacement, x_position_after)

        terminated = self.terminated

        self.num_steps += 1
        truncated = self.num_steps == 1000

        observation = self._get_obs()

        # Compile informative statistics to pass back to the caller
        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "penalty_ctrl": ctrl_cost,
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        # Render the current simulation frame
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Agent is allowed to sense the position and velocity of each DOF across all its joints
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate((position, velocity))

    def reset_model(self):
        # Low and high ends of the random noise that gets added to the initial positions and velocities
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Add noise to the initial positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        # self.set_state(self.init_qpos, self.init_qvel)

        self.num_steps = 0
        self.displacement = self.get_body_com(self.root_body)[0].copy()

        observation = self._get_obs()

        return observation
