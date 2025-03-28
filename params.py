import torch
import sys

sys.path.append("./")
from environments.obstacles import Obstacle
from environments.mobile_robots_env import MobileRobotsEnv
from ddpg_agents.mad_controller import MADController
from ddpg_agents.ma_controller import MAController
from ddpg_agents.ddpg_controller import DDPGController

import warnings

warnings.simplefilter("ignore", UserWarning)

# Defining Obstacles
def init_obstacles():

    obstacle_positions = [
        [-5, 0],
        [-4.5, 0],
        [-4, 0],
        [-3.5, 0],
        [-3, 0],
        [-2.5, 0],
        [-2, 0],
        [-1.5, 0],
        [-1, 0],
        [1, 0],
        [1.5, 0],
        [2.5, 0],
        [2, 0],
        [3, 0],
        [3.5, 0],
        [4, 0],
        [4.5, 0],
        [5, 0],
    ]
    obstacle_radii = [0.25 for i in obstacle_positions]

    Obs = Obstacle(obstacle_positions, obstacle_radii)

    return Obs


# Defining Environment
def init_env():

    agent_1_initial_low = torch.FloatTensor([-3.0, -3.0, 0.0, 0.0])
    agent_2_initial_low = torch.FloatTensor([1.0, -3.0, 0.0, 0.0])
    initial_state_low = torch.cat([agent_1_initial_low, agent_2_initial_low])

    agent_1_initial_high = torch.FloatTensor([-1.0, -1.0, 0.0, 0.0])
    agent_2_initial_high = torch.FloatTensor([3.0, -1.0, 0.0, 0.0])
    initial_state_high = torch.cat([agent_1_initial_high, agent_2_initial_high])

    agent_1_target = torch.FloatTensor([2.0, 2.0, 0.0, 0.0])
    agent_2_target = torch.FloatTensor([-2.0, 2.0, 0.0, 0.0])
    target_positions = torch.cat([agent_1_target, agent_2_target])

    u_lim = 1
    x_lim = 5
    control_limit_low = -torch.ones(4) * u_lim
    control_limit_high = torch.ones(4) * u_lim
    state_limit_low = -torch.ones(8) * x_lim
    state_limit_high = torch.ones(8) * x_lim

    obstacle_avoidance_function = "pdf"

    env = MobileRobotsEnv(
        n_agents=2,
        target_positions=target_positions,
        obstacle=init_obstacles(),
        obstacle_avoidance_loss_function=obstacle_avoidance_function,
        control_reward_regularization=True,
        prestabilized=True,
        disturbance=True,
        nonlinear_damping=True,
        obstacle_avoidance=True,
        collision_avoidance=True,
        initial_state_low=initial_state_low,
        initial_state_high=initial_state_high,
        state_limit_low=state_limit_low,
        state_limit_high=state_limit_high,
        control_limit_low=control_limit_low,
        control_limit_high=control_limit_high,
    )

    env.R *= 0.025
    env.Q *= 1
    env.alpha_obst = 100
    env.alpha_cer = 0.0
    env.alpha_ca = 5

    # Strength of Pre-Stabilization
    env.k = 0.1

    env.reset()
    print("Built Environment.")
    return env


# Defining MAD Controllers
def init_mad_controller():

    u_lim = 1
    agent_1_target = torch.FloatTensor([2.0, 2.0, 0.0, 0.0])
    agent_2_target = torch.FloatTensor([-2.0, 2.0, 0.0, 0.0])
    target_positions = torch.cat([agent_1_target, agent_2_target])

    # Exploration Noise
    noise_std = 0.01

    Controller = MADController(
        env=init_env(),
        buffer_capacity=100000,
        target_state=target_positions,
        num_dynamics_states=16,
        dynamics_input_time_window_length=500,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.002,
        std_dev=noise_std,
        control_action_upper_bound=u_lim,
        control_action_lower_bound=-u_lim,
    )

    print("Initialized MAD Controller.")
    return Controller


def init_ma_controller():

    u_lim = 1
    agent_1_target = torch.FloatTensor([2.0, 2.0, 0.0, 0.0])
    agent_2_target = torch.FloatTensor([-2.0, 2.0, 0.0, 0.0])
    target_positions = torch.cat([agent_1_target, agent_2_target])

    # Exploration Noise
    noise_std = 0.001

    Controller = MAController(
        env=init_env(),
        buffer_capacity=100000,
        target_state=target_positions,
        num_dynamics_states=16,
        dynamics_input_time_window_length=500,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.002,
        std_dev=noise_std,
        control_action_upper_bound=u_lim,
        control_action_lower_bound=-u_lim,
    )

    print("Initialized MA Controller.")
    return Controller


def init_ddpg_controller():

    u_lim = 1
    agent_1_target = torch.FloatTensor([2.0, 2.0, 0.0, 0.0])
    agent_2_target = torch.FloatTensor([-2.0, 2.0, 0.0, 0.0])
    target_positions = torch.cat([agent_1_target, agent_2_target])

    # Exploration Noise
    noise_std = 0.001

    Controller = DDPGController(
        env=init_env(),
        buffer_capacity=100000,
        target_state=target_positions,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.002,
        std_dev=noise_std,
        control_action_upper_bound=u_lim,
        control_action_lower_bound=-u_lim,
    )

    print("Initialized DDPG Controller.")
    return Controller
