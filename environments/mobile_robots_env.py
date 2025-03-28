import torch
import numpy as np
import torch.nn.functional as F

import sys

sys.path.append("../")
from environments.dlti_env import DiscreteTransferFunctionEnv
import environments.loss_functions as lf


class MobileRobotsEnv(DiscreteTransferFunctionEnv):
    """
    A custom multi-agent environment for simulating mobile robots based on discrete-time transfer functions.
    The environment supports features such as obstacle avoidance, collision avoidance, control effort regularization,
    and nonlinear damping.

    Args:
        n_agents (int): The number of mobile robot agents.
        target_positions (list or tensor): The target positions that the agents must reach.
        obstacle (object, optional): An obstacle object for obstacle avoidance calculations. Default is None.
        obstacle_avoidance_loss_function (str, optional): The type of loss function to use for obstacle avoidance.
                                                          Options: "pdf", "barrier_function", "potential_field". Default is "pdf".
        prestabilized (bool, optional): Whether to use a prestabilized control system. Default is True.
        disturbance (bool, optional): Whether to add random disturbances to the system dynamics. Default is False.
        nonlinear_damping (bool, optional): Whether to include nonlinear damping in the system dynamics. Default is True.
        obstacle_avoidance (bool, optional): Whether to enable obstacle avoidance in the reward calculation. Default is True.
        collision_avoidance (bool, optional): Whether to enable collision avoidance between agents. Default is True.
        control_reward_regularization (bool, optional): Whether to penalize changes in control actions. Default is False.
        initial_state_low (tensor, optional): Lower bound for the initial state of the agents. Default is None.
        initial_state_high (tensor, optional): Upper bound for the initial state of the agents. Default is None.
        state_limit_low (tensor, optional): Lower bound for the state limits of the agents. Default is None.
        state_limit_high (tensor, optional): Upper bound for the state limits of the agents. Default is None.
        control_limit_low (tensor, optional): Lower bound for the control input limits. Default is None.
        control_limit_high (tensor, optional): Upper bound for the control input limits. Default is None.
    """

    def __init__(
        self,
        n_agents,
        target_positions,
        obstacle=None,
        obstacle_avoidance_loss_function="pdf",
        prestabilized=True,
        disturbance=False,
        nonlinear_damping=True,
        obstacle_avoidance=True,
        collision_avoidance=True,
        control_reward_regularization=False,
        initial_state_low=None,
        initial_state_high=None,
        state_limit_low=None,
        state_limit_high=None,
        control_limit_low=None,
        control_limit_high=None,
    ):
        # Inherit from DiscreteTransferFunctionEnv
        self.n_agents = n_agents
        self.n = 4 * self.n_agents  # Number of states
        self.m = 2 * self.n_agents  # Number of control inputs
        self.prestabilized = prestabilized
        self.disturbance = disturbance
        self.nonlinear_damping = nonlinear_damping
        self.obstacle = obstacle

        self.t = 0
        self.dt = 0.05  # Time step
        self.mass = 1.0  # Mass
        self.b = 1.0  # Linear drag coefficient
        self.b2 = 0.1 if nonlinear_damping else 0.0  # Nonlinear drag coefficient
        self.k = 1.0 if self.prestabilized else 0.0  # Spring constant

        self.obstacle_avoidance = obstacle_avoidance
        self.collision_avoidance = collision_avoidance

        if self.obstacle_avoidance:
            self.obstacle_avoidance_loss_function = getattr(
                lf, f"loss_obstacle_avoidance_{obstacle_avoidance_loss_function}"
            )

        self.alpha_obst = 1
        self.alpha_ca = 1
        self.alpha_cer = 1

        self.step_reward_state_error = None
        self.step_reward_control_effort = None
        self.step_reward_control_effort_regularization = None
        self.step_reward_obstacle_avoidance = None
        self.step_reward_collision_avoidance = None

        # Target/desired state (equilibrium)
        self.target_positions = torch.tensor(target_positions, dtype=torch.float32)

        # Define state and control limits for the environment
        if initial_state_low is None:
            initial_state_low = -1 * torch.ones((self.n,))
        if initial_state_high is None:
            initial_state_high = 1 * torch.ones((self.n,))
        if control_limit_low is None:
            control_limit_low = -torch.inf * torch.ones((self.m,))
        if control_limit_high is None:
            control_limit_high = torch.inf * torch.ones((self.m,))

        super(MobileRobotsEnv, self).__init__(
            A=None,  # We'll dynamically compute A at each step
            B=None,  # B is initialized based on the robot system
            Q=torch.eye(self.n),  # Default identity cost for simplicity
            R=torch.eye(self.m),  # Control input regularization
            dt=self.dt,  # Time step
            initial_state_low=initial_state_low,
            initial_state_high=initial_state_high,
            state_limit_low=state_limit_low,
            state_limit_high=state_limit_high,
            control_limit_low=control_limit_low,
            control_limit_high=control_limit_high,
        )

        # Initial state
        self.state = None
        self.w = None
        self.prev_action = torch.zeros(self.m)
        self.control_reward_regularization = control_reward_regularization

        # B matrix (Control input influence on the system)
        self.B = self.dt * torch.kron(
            torch.eye(self.n_agents),
            torch.tensor([[0, 0], [0, 0], [1 / self.mass, 0], [0, 1 / self.mass]]),
        )

    def A_matrix(self, x):
        """
        Constructs the system matrix (A matrix) based on the current state of the system, including
        damping, prestabilization, and mass-spring parameters.

        Args:
            x (torch.Tensor): The current state of the system.

        Returns:
            torch.Tensor: The computed A matrix.
        """
        b1 = self.b
        m, k = self.mass, self.k

        A1 = torch.eye(4 * self.n_agents)
        A2 = torch.cat(
            (
                torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1),
                torch.cat(
                    (
                        torch.diag(torch.tensor([-k / m, -k / m])),
                        torch.diag(torch.tensor([-b1 / m, -b1 / m])),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        A2 = torch.kron(torch.eye(self.n_agents), A2)
        A = A1 + self.dt * A2
        return A

    def f(self, x, u):
        """
        State update function that computes the next state of the system based on the current state
        and control inputs using the system's dynamics.

        Args:
            x (torch.Tensor): The current state of the system.
            u (torch.Tensor): The control input vector for the agents.

        Returns:
            torch.Tensor: The next state of the system after applying control inputs.
        """
        A_x = self.A_matrix(x)
        mask = torch.cat([torch.zeros(2), torch.ones(2)]).repeat(self.n_agents)
        if self.prestabilized:
            # State evolution: x_{t+1} = A(x) @ (x - target_positions) + B @ u + target_positions
            f = (
                F.linear(x - self.target_positions, A_x)
                + F.linear(u, self.B)
                + self.dt
                * self.b2
                / self.mass
                * mask
                * torch.tanh(x - self.target_positions)
                + self.target_positions
            )
        else:
            # State evolution: x_{t+1} = A(x) @ x + B @ u
            f = F.linear(x, A_x) + F.linear(u, self.B)
        return f

    def step(self, action):
        """
        Advances the environment by one time step using the provided action. Computes the next state
        of the system and calculates the reward based on various loss functions such as state tracking,
        control effort, obstacle avoidance, and collision avoidance.

        Args:
            action (numpy.ndarray or torch.Tensor): The control action for the agents.

        Returns:
            tuple: A tuple containing the next state, reward, termination flag, truncation flag, and additional info.
        """

        # Convert action_space limits to torch tensors if they are not already
        action_low = torch.tensor(self.action_space.low, dtype=torch.float32)
        action_high = torch.tensor(self.action_space.high, dtype=torch.float32)

        # Convert action to tensor if it's a NumPy array
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)

        # Clamp the action within the allowed range
        action = torch.clamp(action, action_low, action_high)

        # Convert action to torch tensor (in case it isn't already)
        action_t = torch.tensor(action, dtype=torch.float32)

        # Update the state using the dynamics (discrete-time)
        self.state = self.f(torch.tensor(self.state, dtype=torch.float32), action_t)

        if self.disturbance:
            self.w = self.disturbance_process()
            self.state += self.w

        # Calculate the reward (negative loss)
        loss_states = lf.loss_state_tracking(self.state, self.target_positions, self.Q)
        self.step_reward_state_error = -loss_states
        loss_action = lf.loss_control_effort(action_t, self.R)
        self.step_reward_control_effort = -loss_action

        self.step_reward_control_effort_regularization = 0
        if self.control_reward_regularization:
            loss_action_regularized = lf.loss_control_effort_regularized(
                u=action_t, u_prev=self.prev_action, R=self.R
            )
            self.step_reward_control_effort_regularization += -(
                self.alpha_cer * loss_action_regularized
            )
            self.prev_action = action_t

        self.step_reward_obstacle_avoidance = 0
        if self.obstacle_avoidance:
            loss_obst = self.obstacle.get_obstacle_avoidance_loss(
                self.state, self.obstacle_avoidance_loss_function, self.alpha_obst
            )
            self.step_reward_obstacle_avoidance += -loss_obst

        self.step_reward_collision_avoidance = 0
        if self.collision_avoidance:
            loss_ca = self.loss_collision_avoidance()
            self.step_reward_collision_avoidance += -loss_ca

        reward = (
            self.step_reward_state_error
            + self.step_reward_control_effort
            + self.step_reward_control_effort_regularization
            + self.step_reward_obstacle_avoidance
            + self.step_reward_collision_avoidance
        )

        terminated = False
        truncated = False
        self.t += 1
        return self.state, reward, terminated, truncated, {}

    def reset(self):
        """
        Resets the environment to a random initial state.

        Returns:
            torch.Tensor: The initial state of the system after resetting.
        """
        self.state = (
            torch.rand(self.n_states)
            * (self.initial_state_high - self.initial_state_low)
            + self.initial_state_low
        )
        self.prev_action *= 0.0
        self.t = 0
        return self.state

    def disturbance_process(self):
        """
        Generates a random disturbance process that decays over time, simulating external forces
        acting on the mobile robots.

        Returns:
            torch.Tensor: A disturbance vector added to the system dynamics.
        """
        d = 0.1 * torch.randn(self.n)
        d[3::4] *= 0.1
        d[2::4] *= 0.1
        d *= torch.exp(torch.tensor(-0.05 * self.t))
        return d

    def render(self, mode="human"):
        """
        Renders the current state of the environment. Currently, it prints the state to the console.

        Args:
            mode (str, optional): The rendering mode. Default is "human".
        """
        print(f"State: {self.state}")

    def close(self):
        """
        Closes the environment and releases any resources. Placeholder function.
        """
        pass

    def loss_collision_avoidance(self, radius=0.5):
        """
        Computes the loss for collision avoidance between agents based on the minimum safe distance
        between them. Penalizes configurations where agents are too close to each other.

        Args:
            radius (float, optional): The radius representing the safe space around each agent. Default is 0.5.

        Returns:
            torch.Tensor: The collision avoidance loss based on agent proximity.
        """
        min_sec_dist = 2 * radius + 0.25
        # collision avoidance:
        deltaqx = self.state[0::4].repeat(self.n_agents, 1) - self.state[0::4].repeat(
            self.n_agents, 1
        ).transpose(0, 1)
        deltaqy = self.state[1::4].repeat(self.n_agents, 1) - self.state[1::4].repeat(
            self.n_agents, 1
        ).transpose(0, 1)
        distance_sq = deltaqx**2 + deltaqy**2
        mask = torch.logical_not(torch.eye(self.n_agents, dtype=torch.bool))
        loss_ca = (
            1
            / (distance_sq + 1e-3)
            * (distance_sq.detach() < (min_sec_dist**2))
            * mask
        ).sum() / 2
        return self.alpha_ca * loss_ca
