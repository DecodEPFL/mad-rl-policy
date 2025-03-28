import torch
import torch.nn.functional as F


def loss_state_tracking(state, target_positions, Q):
    """
    Computes a quadratic loss penalizing the deviation of the system's state from the target positions.

    Args:
        state (torch.Tensor): The current state of the system.
        target_positions (torch.Tensor): The target positions or desired state.
        Q (torch.Tensor): State weighting matrix for penalizing the deviation.

    Returns:
        torch.Tensor: The total loss based on the state deviation.
    """
    dx = state - target_positions
    return (F.linear(dx, Q) * dx).sum()


def loss_control_effort(u, R):
    """
    Computes the loss associated with minimizing control effort by penalizing large control inputs.

    Args:
        u (torch.Tensor): Control input vector.
        R (torch.Tensor): Weighting matrix for control effort penalty.

    Returns:
        torch.Tensor: The total control effort loss.
    """
    return (F.linear(u, R) * u).sum()


def loss_control_effort_regularized(u, u_prev, R):
    """
    Computes the loss that penalizes the change in control inputs (delta_u) to ensure smooth control actions.

    Args:
        u (torch.Tensor): Current control input vector.
        u_prev (torch.Tensor): Previous control input vector.
        R (torch.Tensor): Weighting matrix for control effort penalty.

    Returns:
        torch.Tensor: The total loss for the change in control inputs.
    """
    delta_u = u - u_prev
    return (F.linear(delta_u, R) * delta_u).sum()


def loss_obstacle_avoidance_potential_field(state, obstacle_position, obstacle_radius):
    """
    Computes the loss for obstacle avoidance using potential fields. The loss increases as the
    state approaches the obstacle's position, based on the distance between the state and the obstacle.

    Args:
        state (torch.Tensor): The current state of the system.
        obstacle_position (torch.Tensor): The position of the obstacle.
        obstacle_radius (float): The radius of influence of the obstacle.

    Returns:
        torch.Tensor: The loss based on proximity to the obstacle.
    """
    influence_radius = 2 * obstacle_radius
    dist_to_obst = torch.norm(state[:2] - obstacle_position)
    loss = 0.0
    if dist_to_obst < influence_radius:
        loss = 0.5 * ((1 / dist_to_obst) - (1 / influence_radius)) ** 2
    return loss


def loss_obstacle_avoidance_barrier_function(state, obstacle_position, obstacle_radius):
    """
    Computes the loss for obstacle avoidance using a barrier function. As the state gets too close
    to the obstacle's radius, the loss becomes significantly large.

    Args:
        state (torch.Tensor): The current state of the system.
        obstacle_position (torch.Tensor): The position of the obstacle.
        obstacle_radius (float): The radius of the obstacle.

    Returns:
        torch.Tensor: The loss based on proximity to the obstacle.
    """
    influence_radius = 2 * obstacle_radius
    dist_to_obst = torch.norm(state[:2] - obstacle_position)
    loss = 0.0
    if dist_to_obst < influence_radius:
        loss = -1 / (dist_to_obst - influence_radius)
    return loss


def loss_obstacle_avoidance_pdf(state, obstacle_position, obstacle_radius):
    """
    Computes the obstacle avoidance loss using a probability density function (PDF) based approach.
    It uses the system's state and the obstacle's position to calculate the probability of being near the obstacle.

    Args:
        state (torch.Tensor): The current state of the system.
        obstacle_position (torch.Tensor): The position of the obstacle.
        obstacle_radius (float): The radius of the obstacle.

    Returns:
        torch.Tensor: The loss based on the PDF of the state near the obstacle.
    """
    qx = state[::4].unsqueeze(1)
    qy = state[1::4].unsqueeze(1)
    q = torch.cat((qx, qy), dim=1).view(1, -1).squeeze()

    mu = obstacle_position
    cov = torch.tensor([[obstacle_radius, obstacle_radius]])
    Q = normpdf(q, mu=mu, cov=cov)
    return Q.sum()


def normpdf(q, mu, cov):
    """
    Computes the multivariate normal probability density function (PDF) for a given input.

    Args:
        q (torch.Tensor): The query point(s) to compute the PDF for.
        mu (torch.Tensor): The mean (center) of the PDF.
        cov (torch.Tensor): The covariance matrix (diagonal entries) of the PDF.

    Returns:
        torch.Tensor: The computed PDF value at the query point.
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2)
    out = torch.tensor(0, dtype=torch.float32)
    for qi in qs:
        den = (2 * torch.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu) ** 2 / cov).sum())
        out = out + num / den
    return out
