import torch


class Obstacle:
    """
    A class to represent obstacles in the environment.

    Args:
        positions (list of tuples): List of (x, y) positions of obstacles.
        radii (list of floats): List of radii for each obstacle.
    """

    def __init__(self, positions, radii):
        """
        Initialize obstacles.

        Args:
            positions (list of tuples): List of (x, y) positions of obstacles.
            radii (list of floats): List of radii for each obstacle.
        """
        assert len(positions) == len(radii), "Each obstacle must have a radius"
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.radii = torch.tensor(radii, dtype=torch.float32)

    def get_obstacles(self):
        """
        Returns the positions and radii of the obstacles.

        Returns:
            tuple: A tuple containing tensors of obstacle positions and radii.
        """
        return self.positions, self.radii

    def get_obstacle_avoidance_loss(self, state, loss_function, alpha_obst, radius=0.5):
        """
        Apply obstacle avoidance loss based on the provided loss function.

        Args:
            state (torch.Tensor): The agent's current state.
            loss_function (function): The loss function to compute obstacle avoidance.
            alpha_obst (float): The weight of the obstacle avoidance loss.
            radius (float, optional): The radius around the obstacle to avoid. Default is 0.5.

        Returns:
            torch.Tensor: The total obstacle avoidance loss for all obstacles.
        """
        loss = torch.tensor(0.0)
        for i in range(self.positions.shape[0]):
            loss += loss_function(state, self.positions[i], self.radii[i])
        return alpha_obst * loss
