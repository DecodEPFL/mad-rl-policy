import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("../")
from ssm.ssm import SSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ornstein-Uhlenbeck noise process
class OUActionNoise:
    """
    Class for implementing the Ornstein-Uhlenbeck noise process, commonly used for exploration in reinforcement learning.

    Attributes:
        mean (numpy.ndarray): The mean of the noise process.
        std_deviation (numpy.ndarray): The standard deviation of the noise process.
        theta (float): The rate of mean reversion, default is 0.15.
        dt (float): Time step used in the process, default is 1e-2.
        x_initial (numpy.ndarray or None): Initial value for the noise process, default is None (sets to zero).
        x_prev (numpy.ndarray): The previous state of the noise process.

    Methods:
        __call__(): Generates the next noise value based on the Ornstein-Uhlenbeck process.
        reset(): Resets the noise process to its initial state.
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        Initializes the Ornstein-Uhlenbeck noise process.

        Args:
            mean (numpy.ndarray): The mean of the noise process.
            std_deviation (numpy.ndarray): The standard deviation of the noise process.
            theta (float, optional): The rate of mean reversion, default is 0.15.
            dt (float, optional): Time step for the process, default is 1e-2.
            x_initial (numpy.ndarray or None, optional): Initial state of the noise process, default is None.
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
        Generates the next value from the Ornstein-Uhlenbeck noise process.

        Returns:
            numpy.ndarray: The generated noise value.
        """
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the noise process to its initial state.

        If no initial state is provided, it is set to zero.
        """
        self.x_prev = (
            self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)
        )


# Replay buffer
class ReplayBuffer:
    """
    A replay buffer for storing and sampling experience tuples used for training reinforcement learning agents.

    Attributes:
        buffer_capacity (int): The maximum number of samples the buffer can store, default is 100000.
        batch_size (int): The number of samples to return in each batch, default is 64.
        dynamics_input_time_window (int): The time window of dynamics input to store, default is 50.
        buffer_counter (int): Counter for the number of samples added.
        num_states (int): The number of state features.
        num_dynamics_states (int): The number of dynamics state features.
        num_actions (int): The number of action features.
        state_buffer (numpy.ndarray): Buffer storing states.
        action_buffer (numpy.ndarray): Buffer storing actions.
        reward_buffer (numpy.ndarray): Buffer storing rewards.
        next_state_buffer (numpy.ndarray): Buffer storing next states.
        initial_state_buffer (numpy.ndarray): Buffer storing initial state time windows.
        next_initial_state_buffer (numpy.ndarray): Buffer storing next initial state time windows.
        dynamics_states_real_buffer (numpy.ndarray): Buffer storing real dynamics states.
        next_dynamics_states_real_buffer (numpy.ndarray): Buffer storing next dynamics states.
        disturbance_buffer (numpy.ndarray): Buffer storing disturbance information.
        next_disturbance_buffer (numpy.ndarray): Buffer storing next disturbance information.

    Methods:
        record(obs_tuple): Records a new experience tuple into the buffer.
        sample(): Samples a batch of experience tuples from the buffer.
    """

    def __init__(
        self,
        buffer_capacity=100000,
        batch_size=64,
        num_states=1,
        num_dynamics_states=1,
        num_actions=1,
        dynamics_input_time_window=50,
    ):
        """
        Initializes the replay buffer with the given parameters.

        Args:
            buffer_capacity (int, optional): The maximum size of the buffer, default is 100000.
            batch_size (int, optional): The number of samples per batch, default is 64.
            num_states (int, optional): Number of state features, default is 1.
            num_dynamics_states (int, optional): Number of dynamics state features, default is 1.
            num_actions (int, optional): Number of action features, default is 1.
            dynamics_input_time_window (int, optional): The time window for storing dynamics input, default is 50.
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.dynamics_input_time_window = dynamics_input_time_window
        self.buffer_counter = 0
        self.num_states = num_states
        self.num_dynamics_states = num_dynamics_states
        self.num_actions = num_actions

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.initial_state_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_states)
        )
        self.next_initial_state_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_states)
        )
        self.dynamics_states_real_buffer = np.zeros(
            (self.buffer_capacity, 1 * self.num_dynamics_states)
        )
        self.next_dynamics_states_real_buffer = np.zeros(
            (self.buffer_capacity, 1 * self.num_dynamics_states)
        )
        self.disturbance_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_states)
        )
        self.next_disturbance_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_states)
        )

    def record(self, obs_tuple):
        """
        Records a new experience tuple into the buffer.

        Args:
            obs_tuple (tuple): A tuple containing state, action, reward, next_state,
                                initial_state, next_initial_state, dynamics states,
                                next dynamics states, disturbance, and next disturbance.
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.initial_state_buffer[index] = obs_tuple[4]
        self.next_initial_state_buffer[index] = obs_tuple[5]
        self.dynamics_states_real_buffer[index] = obs_tuple[6]
        self.next_dynamics_states_real_buffer[index] = obs_tuple[7]
        self.disturbance_buffer[index] = obs_tuple[8]
        self.next_disturbance_buffer[index] = obs_tuple[9]

        self.buffer_counter += 1

    def sample(self):
        """
        Samples a batch of experience tuples from the buffer.

        Returns:
            tuple: A tuple of tensors representing the sampled experience batch.
        """
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = torch.tensor(
            self.state_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        action_batch = torch.tensor(
            self.action_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        reward_batch = torch.tensor(
            self.reward_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        next_state_batch = torch.tensor(
            self.next_state_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        dynamics_input_time_window_batch = torch.tensor(
            self.initial_state_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        next_dynamics_input_time_window_batch = torch.tensor(
            self.next_initial_state_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        dynamics_states_real_batch = torch.tensor(
            self.dynamics_states_real_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        next_dynamics_states_real_batch = torch.tensor(
            self.next_dynamics_states_real_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        dynamics_disturbance_time_window_batch = torch.tensor(
            self.disturbance_buffer[batch_indices], dtype=torch.float32
        ).to(device)
        next_dynamics_disturbance_time_window_batch = torch.tensor(
            self.next_disturbance_buffer[batch_indices], dtype=torch.float32
        ).to(device)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            dynamics_input_time_window_batch,
            next_dynamics_input_time_window_batch,
            dynamics_states_real_batch,
            next_dynamics_states_real_batch,
            dynamics_disturbance_time_window_batch,
            next_dynamics_disturbance_time_window_batch,
        )


# Actor Network
class Actor(nn.Module):
    """
    The Actor network that takes the state and predicts the control action.

    Attributes:
        control_action_upper_bound (float): The upper bound for the control action.
        direction_mlp (torch.nn.Sequential): MLP used to predict the direction term of the action.
        m_dynamics (SSM): A state-space model used to handle dynamics.
        num_states (int): Number of input states.
        m_term (torch.Tensor or None): The model term of the action.
        a_term (torch.Tensor or None): The action term.
        d_term (torch.Tensor or None): The disturbance term.

    Methods:
        _initialize_weights(): Initializes the weights and biases of the network.
        forward(state, dynamics_input_time_window, dynamics_disturbance_time_window):
            Forward pass to compute the action.
    """

    def __init__(
        self,
        num_states,
        num_actions,
        control_action_upper_bound,
        num_dynamics_states=2,
        hidden_dim=16,
    ):
        """
        Initializes the Actor network.

        Args:
            num_states (int): Number of input states.
            num_actions (int): Number of output actions.
            control_action_upper_bound (float): The upper bound for the control action.
            num_dynamics_states (int, optional): Number of dynamics states, default is 2.
            hidden_dim (int, optional): The number of hidden units in the MLP, default is 16.
        """
        super(Actor, self).__init__()
        self.control_action_upper_bound = control_action_upper_bound

        self.m_dynamics = SSM(
            in_features=num_states,
            out_features=num_actions,
            state_features=num_dynamics_states,
            scan=True,
        )

        self.num_states = num_states
        self.m_term = None
        self.a_term = None
        self.d_term = None

    def forward(
        self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Forward pass to compute the action based on the state and dynamics input.

        Args:
            state (torch.Tensor): The current state.
            dynamics_input_time_window (torch.Tensor): The initial state time window.
            dynamics_disturbance_time_window (torch.Tensor): The disturbance time window.

        Returns:
            torch.Tensor: The computed action.
        """

        if dynamics_disturbance_time_window.dim() == 2:
            dynamics_disturbance = dynamics_disturbance_time_window.unsqueeze(0)
            dynamics_disturbance_output = self.m_dynamics(dynamics_disturbance)
            self.m_term = dynamics_disturbance_output.squeeze()[-1, :]
            action = self.m_term
            action = action.squeeze(0)

        elif dynamics_disturbance_time_window.dim() == 3:
            dynamics_disturbance = dynamics_disturbance_time_window
            dynamics_disturbance_output = self.m_dynamics(dynamics_disturbance)
            m_term = dynamics_disturbance_output.squeeze()[:, -1, :]
            action = m_term
            action = action.squeeze(0)

        return action


# Critic network
class Critic(nn.Module):
    """
    The Critic network used to estimate the value function.

    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        fc3 (torch.nn.Linear): Third fully connected layer.
        fc4 (torch.nn.Linear): Fourth fully connected layer.
        fc5 (torch.nn.Linear): Fifth fully connected layer.
        fc6 (torch.nn.Linear): Final output layer.

    Methods:
        forward(state, dynamics_states_real, action): Forward pass to compute the value estimate.
    """

    def __init__(self, num_states, num_dynamics_states, num_actions):
        """
        Initializes the Critic network.

        Args:
            num_states (int): Number of input states.
            num_dynamics_states (int): Number of dynamics states.
            num_actions (int): Number of actions.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_states + 1 * num_dynamics_states, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(num_actions, 32)
        self.fc4 = nn.Linear(32 * 2, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, dynamics_states_real, action):
        """
        Forward pass to compute the value estimate.

        Args:
            state (torch.Tensor): The state tensor.
            dynamics_states_real (torch.Tensor): The real dynamics states tensor.
            action (torch.Tensor): The action tensor.

        Returns:
            torch.Tensor: The computed value estimate.
        """
        state_augmented = torch.cat([state, 2 * dynamics_states_real], dim=-1)
        state_out = F.tanh(self.fc1(state_augmented))
        state_out = F.tanh(self.fc2(state_out))
        action_out = F.tanh(self.fc3(action))
        concat = torch.cat([state_out, action_out], dim=-1)
        x = F.relu(self.fc4(concat))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# DDPG agent class
class MAController:
    """
    MAController class for implementing a Magnitude And Direction Policy Parametrization (without Direction term) based control agent using Deep Deterministic Policy Gradient (DDPG) algorithm.

    This class manages the agent's interaction with an environment, its policy, and learning process. It includes functionality for managing dynamics input and disturbance windows, actor and critic models, and updating target networks. The agent operates in continuous action spaces and is trained using DDPG with experience replay and Ornstein-Uhlenbeck noise for exploration.

    Attributes:
        env (gym.Env): The environment with which the agent interacts.
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions in the environment.
        state (torch.Tensor): Current state of the agent.
        target_state (torch.Tensor): Target state the agent is aiming for.
        state_error (torch.Tensor): Difference between current state and target state.
        w (torch.Tensor): State-related dynamics information.
        num_dynamics_states (int): The number of dynamics states.
        dynamics_states_real (torch.Tensor): Real-time dynamics states.
        dynamics_input_time_window (torch.Tensor): The window of recent dynamics inputs.
        dynamics_disturbance_time_window (torch.Tensor): The window of recent dynamics disturbances.
        ep_initial_state (torch.Tensor): Initial state for the episode.
        ep_timestep (torch.Tensor): Current timestep in the episode.
        actor_model (Actor): The actor network, responsible for selecting actions.
        target_actor (Actor): The target actor network, used for stable updates.
        critic_model (Critic): The critic network, evaluates actions taken by the actor.
        target_critic (Critic): The target critic network, used for stable updates.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        ou_noise (OUActionNoise): Ornstein-Uhlenbeck noise for exploration.
        buffer (ReplayBuffer): Experience replay buffer for training.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update parameter for target networks.
        control_action_upper_bound (float): Upper bound for the action values.
        control_action_lower_bound (float): Lower bound for the action values.
        episode_count (int): Counter for the number of episodes.
        rewards_list (list): List of rewards accumulated during training.
        rewards_ma50_list (list): List of 50-episode moving average rewards.

    Methods:
        __init__: Initializes the MADController with the provided parameters.
        set_ep_initial_state: Sets the initial state for the episode relative to the target state.
        update_dynamics_input_time_window: Updates the dynamics input time window with the current state.
        reset_ep_timestep: Resets the episode timestep and clears the disturbance window.
        update_ep_timestep: Increments the episode timestep and updates the state error.
        policy: Selects an action based on the current state and dynamics information.
        learned_policy: Selects an action using the learned policy (actor model).
        update_target: Soft updates the target actor and target critic networks.
        train: Trains the agent over multiple episodes using the DDPG algorithm.
        get_trajectory: Generates a trajectory by interacting with the environment from a given initial state.
        get_trajectory_with_loss_terms: Generates a trajectory and collects loss terms for various components.
        save_model_weights: Saves the weights of the actor, critic, and optimizers to a file.
        load_model_weight: Loads the weights of the actor, critic, and optimizers from a file.
    """

    def __init__(
        self,
        env,
        buffer_capacity=100000,
        target_state=None,
        num_dynamics_states=2,
        dynamics_input_time_window_length=500,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.002,
        std_dev=0.2,
        control_action_upper_bound=1,
        control_action_lower_bound=-1,
    ):
        """
        Initializes the MAController agent for training in a given environment.

        Args:
            env: The environment to interact with (should support OpenAI Gym interface).
            buffer_capacity (int, optional): The capacity of the replay buffer. Defaults to 100000.
            target_state (array-like, optional): The target state for the agent. Defaults to None.
            num_dynamics_states (int, optional): The number of dynamics states for the agent. Defaults to 2.
            dynamics_input_time_window_length (int, optional): The length of the dynamics input time window. Defaults to 500.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
            tau (float, optional): The soft update factor for target networks. Defaults to 0.005.
            actor_lr (float, optional): The learning rate for the actor model. Defaults to 0.001.
            critic_lr (float, optional): The learning rate for the critic model. Defaults to 0.002.
            std_dev (float, optional): The standard deviation for noise in action selection. Defaults to 0.2.
            control_action_upper_bound (float, optional): The upper bound for control actions. Defaults to 1.
            control_action_lower_bound (float, optional): The lower bound for control actions. Defaults to -1.
        """

        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.state = torch.zeros(self.num_states).to(device)
        if target_state is None:
            self.target_state = torch.zeros(self.num_states).to(device)
        else:
            self.target_state = torch.tensor(target_state).to(device)
        self.state_error = self.state - self.target_state
        self.w = torch.zeros(self.num_states).to(device)

        self.num_dynamics_states = num_dynamics_states
        self.dynamics_states_real = torch.zeros(1 * self.num_dynamics_states).to(device)
        self.dynamics_input_time_window_length = dynamics_input_time_window_length
        self.dynamics_input_time_window = torch.zeros(
            (self.dynamics_input_time_window_length, self.num_states)
        ).to(device)
        self.dynamics_disturbance_time_window = torch.zeros(
            (self.dynamics_input_time_window_length, self.num_states)
        ).to(device)
        self.ep_initial_state = torch.zeros(self.num_states).to(device)
        self.ep_timestep = torch.ones(1).to(device)

        self.actor_model = Actor(
            self.num_states,
            self.num_actions,
            control_action_upper_bound,
            self.num_dynamics_states,
        ).to(device)
        self.target_actor = Actor(
            self.num_states,
            self.num_actions,
            control_action_upper_bound,
            self.num_dynamics_states,
        ).to(device)
        self.critic_model = Critic(
            self.num_states, self.num_dynamics_states, self.num_actions
        ).to(device)
        self.target_critic = Critic(
            self.num_states, self.num_dynamics_states, self.num_actions
        ).to(device)

        self.target_actor.load_state_dict(self.actor_model.state_dict())
        self.target_critic.load_state_dict(self.critic_model.state_dict())

        self.critic_optimizer = optim.AdamW(
            self.critic_model.parameters(), lr=critic_lr
        )
        self.actor_optimizer = optim.AdamW(self.actor_model.parameters(), lr=actor_lr)

        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.num_actions),
            std_deviation=float(std_dev) * np.ones(self.num_actions),
        )
        self.buffer = ReplayBuffer(
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            num_states=self.num_states,
            num_dynamics_states=self.num_dynamics_states,
            num_actions=self.num_actions,
            dynamics_input_time_window=self.dynamics_input_time_window_length,
        )

        self.gamma = gamma
        self.tau = tau
        self.control_action_upper_bound = control_action_upper_bound
        self.control_action_lower_bound = control_action_lower_bound

        self.episode_count = 0
        self.rewards_list = []
        self.rewards_ma50_list = []

    def set_ep_initial_state(self, initial_state):
        """
        Sets the initial state of the episode relative to the target state.

        Args:
            initial_state (array-like): The initial state for the episode.
        """
        self.ep_initial_state = torch.tensor(initial_state, dtype=torch.float32).to(
            device
        )
        self.ep_initial_state = self.ep_initial_state - self.target_state

    def update_dynamics_input_time_window(self):
        """
        Updates the dynamics input time window with the current state.
        """
        self.dynamics_input_time_window *= 0.0
        update_index = int(
            self.dynamics_input_time_window_length - self.ep_timestep.cpu().item()
        )
        if update_index >= 0:
            self.dynamics_input_time_window[update_index] = self.ep_initial_state
        if self.ep_timestep == 1:
            self.dynamics_disturbance_time_window[-1] = self.ep_initial_state
        if self.ep_timestep >= self.dynamics_input_time_window_length:
            self.dynamics_disturbance_time_window = (
                self.dynamics_disturbance_time_window
            )
        else:
            temp = torch.roll(self.dynamics_disturbance_time_window, shifts=-1, dims=0)
            temp[-1] = self.w
            self.dynamics_disturbance_time_window = temp

    def reset_ep_timestep(self):
        """
        Resets the episode timestep and clears the dynamics disturbance time window.
        """
        self.ep_timestep = torch.ones(1).to(device)
        self.state_error = self.state - self.target_state
        self.dynamics_disturbance_time_window *= 0.0

    def update_ep_timestep(self):
        """
        Increments the episode timestep and updates the state error.
        """
        self.ep_timestep += 1
        self.state_error = self.state - self.target_state

    def policy(
        self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Selects an action based on the current state and dynamics information, adding noise for exploration.

        Args:
            state (array-like): The current state of the agent.
            dynamics_input_time_window (array-like): The current dynamics input time window.
            dynamics_disturbance_time_window (array-like): The current dynamics disturbance time window.

        Returns:
            np.ndarray: The selected action within the valid control bounds.
        """
        state = torch.tensor(state, dtype=torch.float32).to(device)
        sampled_actions = (
            self.actor_model(
                state, dynamics_input_time_window, dynamics_disturbance_time_window
            )
            .cpu()
            .detach()
            .numpy()
        )
        noise = self.ou_noise()
        sampled_actions += noise
        legal_action = np.clip(
            sampled_actions,
            self.control_action_lower_bound,
            self.control_action_upper_bound,
        )
        return legal_action

    def learned_policy(
        self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Selects an action using the learned policy (actor model).

        Args:
            state (array-like): The current state of the agent.
            dynamics_input_time_window (array-like): The current dynamics input time window.
            dynamics_disturbance_time_window (array-like): The current dynamics disturbance time window.

        Returns:
            np.ndarray: The selected action within the valid control bounds.
        """
        state = torch.tensor(state, dtype=torch.float32).to(device)
        sampled_actions = (
            self.actor_model(
                state, dynamics_input_time_window, dynamics_disturbance_time_window
            )
            .cpu()
            .detach()
            .numpy()
        )
        legal_action = np.clip(
            sampled_actions,
            self.control_action_lower_bound,
            self.control_action_upper_bound,
        )
        return legal_action

    def update_target(self):
        """
        Soft updates the target actor and target critic networks based on the current actor and critic networks using the tau parameter.
        """
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, total_episodes=100, episode_length=100, verbose=True):
        """
        Trains the agent over multiple episodes using the DDPG algorithm.

        Args:
            total_episodes (int, optional): The total number of episodes to train the agent. Defaults to 100.
            episode_length (int, optional): The maximum length of each episode. Defaults to 100.
            verbose (bool, optional): If True, prints episode details during training. Defaults to True.
        """

        for ep in range(total_episodes):
            self.episode_count += 1
            self.ou_noise.reset()
            self.state = torch.tensor(self.env.reset(), dtype=torch.float32).to(device)
            self.dynamics_states_real = torch.cat(
                [
                    self.actor_model.m_dynamics.LRUR.states_last.real.squeeze(),
                ]
            )
            self.set_ep_initial_state(initial_state=self.state)
            self.reset_ep_timestep()
            self.update_dynamics_input_time_window()
            episodic_reward = 0

            while True:
                action = self.policy(
                    state=self.state_error,
                    dynamics_input_time_window=self.dynamics_input_time_window,
                    dynamics_disturbance_time_window=self.dynamics_disturbance_time_window,
                )
                old_state_error = self.state_error.clone()
                old_dynamics_states_real = self.dynamics_states_real.clone()
                old_dynamics_input_time_window = self.dynamics_input_time_window.clone()
                old_dynamics_disturbance_time_window = (
                    self.dynamics_disturbance_time_window.clone()
                )
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.state = torch.tensor(next_state, dtype=torch.float32).to(device)
                self.w = torch.tensor(self.env.w)
                self.dynamics_states_real = torch.cat(
                    [
                        self.actor_model.m_dynamics.LRUR.states_last.real.squeeze(),
                    ]
                )
                self.update_ep_timestep()
                self.update_dynamics_input_time_window()

                obs_tuple = (
                    old_state_error.cpu(),
                    action,
                    reward,
                    self.state_error.cpu(),
                    old_dynamics_input_time_window.cpu(),
                    self.dynamics_input_time_window.cpu(),
                    old_dynamics_states_real.cpu().detach(),
                    self.dynamics_states_real.cpu().detach(),
                    old_dynamics_disturbance_time_window.cpu(),
                    self.dynamics_disturbance_time_window.cpu(),
                )

                self.buffer.record(obs_tuple=obs_tuple)
                episodic_reward += reward

                if self.buffer.buffer_counter > self.buffer.batch_size:
                    (
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        dynamics_input_time_window_batch,
                        next_dynamics_input_time_window_batch,
                        dynamics_states_real_batch,
                        next_dynamics_states_real_batch,
                        dynamics_disturbance_time_window_batch,
                        next_dynamics_disturbance_time_window_batch,
                    ) = self.buffer.sample()

                    with torch.no_grad():
                        target_actions = self.target_actor(
                            state=next_state_batch,
                            dynamics_input_time_window=next_dynamics_input_time_window_batch,
                            dynamics_disturbance_time_window=next_dynamics_disturbance_time_window_batch,
                        )
                        y = reward_batch + self.gamma * self.target_critic(
                            state=next_state_batch,
                            dynamics_states_real=next_dynamics_states_real_batch,
                            action=target_actions,
                        )

                    self.critic_optimizer.zero_grad()
                    critic_value = self.critic_model(
                        state=state_batch,
                        dynamics_states_real=dynamics_states_real_batch,
                        action=action_batch,
                    )
                    critic_loss = F.mse_loss(critic_value, y)
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    self.actor_optimizer.zero_grad()
                    actions = self.actor_model(
                        state=state_batch,
                        dynamics_input_time_window=dynamics_input_time_window_batch,
                        dynamics_disturbance_time_window=dynamics_disturbance_time_window_batch,
                    )
                    actor_loss = -self.critic_model(
                        state=state_batch,
                        dynamics_states_real=dynamics_states_real_batch,
                        action=actions,
                    ).mean()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.update_target()

                if done or truncated or (self.ep_timestep >= episode_length):
                    break

            self.rewards_list.append(episodic_reward)
            ma50_rerward = np.mean(self.rewards_list[-50:])
            self.rewards_ma50_list.append(ma50_rerward)
            if verbose:
                print(
                    f"Episode {self.episode_count}: Ep. Reward = {episodic_reward}, Avg. Reward = {ma50_rerward}"
                )

    def get_trajectory(self, initial_state, timesteps=500):
        """
        Generates a trajectory by interacting with the environment from a given initial state.

        Args:
            initial_state (array-like): The starting state of the trajectory.
            timesteps (int, optional): The number of timesteps to simulate. Defaults to 500.

        Returns:
            tuple: A tuple containing:
                - rewards_list: List of rewards received during the trajectory.
                - obs_list: List of observed states during the trajectory.
                - action_list: List of actions taken during the trajectory.
                - w_list: List of environment states (or any relevant data for the trajectory).
        """

        _ = self.env.reset()
        self.env.state = initial_state
        self.state = initial_state.to(device)
        self.set_ep_initial_state(initial_state=initial_state)
        self.reset_ep_timestep()
        self.update_dynamics_input_time_window()
        obs_list = [initial_state]
        w_list = [initial_state]
        rewards_list = []
        action_list = []

        for _ in range(timesteps):
            state_error = torch.tensor(
                self.env.state.to(device) - self.target_state
            ).to(device)
            action = self.learned_policy(
                state_error,
                self.dynamics_input_time_window,
                self.dynamics_disturbance_time_window,
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.w = torch.tensor(self.env.w)
            obs_list.append(obs)
            w_list.append(self.env.w)
            action_list.append(action)
            rewards_list.append(reward)
            self.update_ep_timestep()
            self.update_dynamics_input_time_window()

        return rewards_list, obs_list, action_list, w_list

    def get_trajectory_with_loss_terms(self, initial_state, timesteps=500):
        """
        Generates a trajectory by interacting with the environment and collects loss terms for various components.

        Args:
            initial_state (array-like): The starting state of the trajectory.
            timesteps (int, optional): The number of timesteps to simulate. Defaults to 500.

        Returns:
            tuple: A tuple containing:
                - rewards_list: List of rewards received during the trajectory.
                - obs_list: List of observed states during the trajectory.
                - action_list: List of actions taken during the trajectory.
                - w_list: List of environment states (or any relevant data for the trajectory).
                - rewards_se: List of state error rewards.
                - rewards_ce: List of control effort rewards.
                - rewards_cer: List of control effort regularization rewards.
                - rewards_oa: List of obstacle avoidance rewards.
                - rewards_ca: List of collision avoidance rewards.
                - distance_list: List of distances between agents during the trajectory.
        """

        _ = self.env.reset()
        self.env.state = initial_state
        self.state = initial_state.to(device)
        self.set_ep_initial_state(initial_state=initial_state)
        self.reset_ep_timestep()
        self.update_dynamics_input_time_window()
        obs_list = [initial_state]
        w_list = [initial_state]
        rewards_list = []
        action_list = []
        rewards_se = []
        rewards_ce = []
        rewards_cer = []
        rewards_oa = []
        rewards_ca = []
        distance_list = []

        for _ in range(timesteps):
            state_error = torch.tensor(
                self.env.state.to(device) - self.target_state
            ).to(device)
            action = self.learned_policy(
                state_error,
                self.dynamics_input_time_window,
                self.dynamics_disturbance_time_window,
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.w = torch.tensor(self.env.w)
            obs_list.append(obs)
            w_list.append(self.env.w)
            action_list.append(action)
            rewards_list.append(reward)
            rewards_se.append(self.env.step_reward_state_error)
            rewards_ce.append(self.env.step_reward_control_effort)
            rewards_cer.append(self.env.step_reward_control_effort_regularization)
            rewards_oa.append(self.env.step_reward_obstacle_avoidance)
            rewards_ca.append(self.env.step_reward_collision_avoidance)
            deltaqx = self.env.state[0::4].repeat(
                self.env.n_agents, 1
            ) - self.env.state[0::4].repeat(self.env.n_agents, 1).transpose(0, 1)
            deltaqy = self.env.state[1::4].repeat(
                self.env.n_agents, 1
            ) - self.env.state[1::4].repeat(self.env.n_agents, 1).transpose(0, 1)
            distance_sq = deltaqx**2 + deltaqy**2
            distance_list.append(np.sqrt(distance_sq[0, 1].cpu()))
            self.update_ep_timestep()
            self.update_dynamics_input_time_window()

        return (
            rewards_list,
            obs_list,
            action_list,
            w_list,
            rewards_se,
            rewards_ce,
            rewards_cer,
            rewards_oa,
            rewards_ca,
            distance_list,
        )

    def save_model_weights(self, filename):
        """
        Saves the weights of the model (actor, critic, and optimizers) to a file.

        Args:
            filename (str): The path to save the model weights.
        """
        state_dict_list = [
            self.actor_model.state_dict(),
            self.critic_model.state_dict(),
            self.target_actor.state_dict(),
            self.target_critic.state_dict(),
            self.actor_optimizer.state_dict(),
            self.critic_optimizer.state_dict(),
            self.actor_model.m_dynamics.state_dict(),
            self.target_actor.m_dynamics.state_dict(),
        ]
        torch.save(state_dict_list, filename)
        print(f"DDPG Model weights saved at {filename}.")

    def load_model_weight(self, filename):
        """
        Loads the weights of the model (actor, critic, and optimizers) from a file.

        Args:
            filename (str): The path to load the model weights from.
        """
        self.actor_model.load_state_dict(torch.load(filename)[0])
        self.critic_model.load_state_dict(torch.load(filename)[1])
        self.target_actor.load_state_dict(torch.load(filename)[2])
        self.target_critic.load_state_dict(torch.load(filename)[3])
        self.actor_optimizer.load_state_dict(torch.load(filename)[4])
        self.critic_optimizer.load_state_dict(torch.load(filename)[5])
        self.actor_model.m_dynamics.load_state_dict(torch.load(filename)[6])
        self.target_actor.m_dynamics.load_state_dict(torch.load(filename)[7])
        print(f"DDPG Model weights loaded from {filename} successfully.")
