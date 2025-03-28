"""
Implements a State-Space Model (SSM) combining various components, including:
- MLP (Multilayer Perceptron): This should be replaced with a Lipschitz-bounded network as per the Manchester model.
- Parallel Scan: A fast implementation designed to speed up implementations.
- LRU (Linear Recurrent Units): A linear system parameterized to ensure stability. See the reference paper: 
  https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf.
- SSM (State-Space Model): A combination of LRU and MLP components. An optional feedforward linear layer can be included
  to map the input directly to the output.
"""

import math
import torch
import torch.nn as nn
from sympy import false


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) network with one input layer,
    one hidden layer, and an output layer. The activation functions
    used are SiLU and ReLU.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output features.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=false),  # First layer
            nn.SiLU(),  # Activation after the first layer
            nn.Linear(hidden_size, hidden_size, bias=false),  # Hidden layer
            nn.ReLU(),  # Activation after hidden layer
            nn.Linear(
                hidden_size, output_size, bias=false
            ),  # Output layer (no activation)
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (tensor): Input tensor of shape (batch_size, input_size) or (batch_size, sequence_length, input_size).

        Returns:
            tensor: Output of the MLP with shape (batch_size, output_size) or (batch_size, sequence_length, output_size).
        """
        if x.dim() == 3:
            # x is of shape (batch_size, sequence_length, input_size)
            batch_size, seq_length, input_size = x.size()

            # Flatten the batch and sequence dimensions for the MLP
            x = x.reshape(-1, input_size)  # Use reshape instead of view

            # Apply the MLP to each feature vector
            x = self.model(x)  # Shape: (batch_size * sequence_length, output_size)

            # Reshape back to (batch_size, sequence_length, output_size)
            output_size = x.size(-1)
            x = x.reshape(
                batch_size, seq_length, output_size
            )  # Use reshape instead of view
        else:
            # If x is not 3D, just apply the MLP directly
            x = self.model(x)
        return x


class PScan(torch.autograd.Function):
    """
    Implements the Parallel Scan algorithm for fast sequential computation in logarithmic time.

    Methods:
        expand_(A, X): Expands A and X in-place to facilitate efficient sequential computation.
        acc_rev_(A, X): Performs reverse accumulation to compute backward pass in scan algorithm.
        forward(ctx, A, X, Y_init): Performs forward pass of the scan algorithm.
        backward(ctx, grad_output): Computes gradients for the backward pass using reverse accumulation.
    """

    @staticmethod
    def expand_(A, X):
        """
        In-place expansion of A and X to allow sequential computation.

        Args:
            A (tensor): Tensor containing the sequence of scalars.
            X (tensor): Tensor containing the sequence of vectors.
        """
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])

    @staticmethod
    def acc_rev_(A, X):
        """
        Reverse accumulation step for gradient computation in the scan algorithm.

        Args:
            A (tensor): Tensor containing the sequence of scalars.
            X (tensor): Tensor containing the sequence of vectors.
        """
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    @staticmethod
    def forward(ctx, A, X, Y_init):
        """
        Forward pass of the parallel scan algorithm.

        Args:
            A (tensor): Tensor of shape (N, T) representing the sequence of scalars.
            X (tensor): Tensor of shape (N, T, D) representing the sequence of vectors.
            Y_init (tensor): Initial state of shape (N, D).

        Returns:
            tensor: Output sequence of shape (N, T, D).
        """
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the parallel scan algorithm.

        Args:
            grad_output (tensor): Gradients of the output sequence.

        Returns:
            tuple: Gradients for inputs A, X, and Y_init.
        """
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)


pscan = PScan.apply


class LRU(nn.Module):
    """
    Implements a Linear Recurrent Unit (LRU) with a stable parametrization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        state_features (int): Number of state features (hidden dimension).
        scan (bool, optional): Whether to use parallel scan for fast computation (default: True).
        rmin (float, optional): Minimum radius for eigenvalues (default: 0.9).
        rmax (float, optional): Maximum radius for eigenvalues (default: 1.0).
        max_phase (float, optional): Maximum phase for eigenvalues (default: 6.283).
    """

    def __init__(
        self,
        in_features,
        out_features,
        state_features,
        scan=True,
        rmin=0.9,
        rmax=1,
        max_phase=6.283,
    ):
        super().__init__()
        self.state_features = state_features
        self.in_features = in_features
        self.scan = scan
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        # self.theta_log = torch.log(max_phase * u2).to(self.nu_log.device)
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.register_buffer(
            "state",
            torch.complex(
                torch.zeros(state_features),  # + 0.1 * torch.rand(state_features),
                torch.zeros(state_features),  # + 0.1 * torch.rand(state_features),
            ),
        )
        self.states_last = self.state

    def forward(self, input):
        """
        Forward pass of the LRU.

        Args:
            input (tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            tensor: Output of the LRU of shape (batch_size, sequence_length, output_size).
        """
        self.state = self.state
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)  # Eigenvalues matrix
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )
        # Input must be (Batches,Seq_length, Input size), otherwise adds dummy dimension = 1 for batches
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if self.scan:  # Simulate the LRU with Parallel Scan
            input = input.permute(2, 1, 0)  # (Input size,Seq_length, Batches)
            # Unsqueeze b to make its shape (N, V, 1, 1)
            B_unsqueezed = self.B.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast b along dimensions T and D so it can be multiplied elementwise with u
            B_broadcasted = B_unsqueezed.expand(
                self.state_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Expand u so that it can be multiplied along dimension N, resulting in shape (N, V, T, D)
            input_broadcasted = input.unsqueeze(0).expand(
                self.state_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            inputBU = torch.sum(
                B_broadcasted * input_broadcasted, dim=1
            )  # (State size,Seq_length, Batches)

            # Prepare matrix Lambda for scan
            Lambda = Lambda.unsqueeze(1)
            A = torch.tile(Lambda, (1, inputBU.shape[1]))
            # Initial condition
            init = torch.complex(
                torch.zeros(
                    (self.state_features, inputBU.shape[2]), device=self.B.device
                ),
                torch.zeros(
                    (self.state_features, inputBU.shape[2]), device=self.B.device
                ),
            )

            gammas_reshaped = gammas.unsqueeze(2)  # Shape becomes (State size, 1, 1)
            # Element-wise multiplication
            GBU = gammas_reshaped * inputBU

            states = pscan(A, GBU, init)  # dimensions: (State size,Seq_length, Batches)
            if states.shape[-1] == 1:
                self.states_last = states.clone().permute(2, 1, 0)[:, -1, :]

            # Prepare output matrices C and D for sequence and batch handling
            # Unsqueeze C to make its shape (Y, X, 1, 1)
            C_unsqueezed = self.C.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            C_broadcasted = C_unsqueezed.expand(
                self.out_features,
                self.state_features,
                inputBU.shape[1],
                inputBU.shape[2],
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            CX = torch.sum(C_broadcasted * states, dim=1)

            # Unsqueeze D to make its shape (Y, U, 1, 1)
            D_unsqueezed = self.D.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            D_broadcasted = D_unsqueezed.expand(
                self.out_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            DU = torch.sum(D_broadcasted * input, dim=1)

            output = 2 * CX.real + DU
            output = output.permute(
                2, 1, 0
            )  # Back to (Batches, Seq length, Input size)
        else:  # Simulate the LRU recursively
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = Lambda * self.state + gammas * self.B @ step.to(
                        dtype=self.B.dtype
                    )
                    out_step = (self.C @ self.state).real + self.D @ step
                    out_seq[j] = out_step
                self.state = torch.complex(
                    torch.zeros_like(self.state.real), torch.zeros_like(self.state.real)
                )
                output[i] = out_seq
        return output  # Shape (Batches,Seq_length, Input size)


class SSM(nn.Module):
    """
    Implements a Structured State Space Model (SSM), combining LRU with an MLP and a linear skip connection.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        state_features (int): Number of state features (hidden dimension).
        scan (bool): Whether to use parallel scan for the LRU.
        mlp_hidden_size (int, optional): Number of hidden units in the MLP (default: 30).
        rmin (float, optional): Minimum radius for LRU eigenvalues (default: 0.9).
        rmax (float, optional): Maximum radius for LRU eigenvalues (default: 1.0).
        max_phase (float, optional): Maximum phase for LRU eigenvalues (default: 6.283).
    """

    def __init__(
        self,
        in_features,
        out_features,
        state_features,
        scan,
        mlp_hidden_size=30,
        rmin=0.9,
        rmax=1,
        max_phase=6.283,
    ):
        super().__init__()
        self.mlp = MLP(out_features, mlp_hidden_size, out_features)
        self.LRUR = LRU(
            in_features, out_features, state_features, scan, rmin, rmax, max_phase
        )
        self.model = nn.Sequential(self.LRUR, self.mlp)
        self.lin = nn.Linear(in_features, out_features, bias=false)

    def set_paramS(self):
        """
        Sets parameters for the LRU block.
        """
        self.LRUR.set_param()

    def forward(self, input):
        """
        Forward pass of the SSM.

        Args:
            input (tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            tensor: Output tensor of shape (batch_size, sequence_length, output_size).
        """
        result = self.model(input) + self.lin(input)
        return result


class DeepLRU(nn.Module):
    """
    Implements a deep model with multiple layers of SSM blocks.

    Args:
        N (int): Number of SSM layers.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        mid_features (int): Number of intermediate features (hidden dimension).
        state_features (int): Number of state features for each SSM.
        scan (bool, optional): Whether to use parallel scan for the LRU (default: True).

    Methods:
        set_param(): Sets parameters for all SSM layers.
        forward(input): Forward pass through the deep model.
    """

    def __init__(
        self, N, in_features, out_features, mid_features, state_features, scan=True
    ):
        super().__init__()
        self.linin = nn.Linear(in_features, mid_features, bias=false)
        self.linout = nn.Linear(mid_features, out_features, bias=false)
        self.modelt = nn.ModuleList(
            [SSM(mid_features, mid_features, state_features, scan) for j in range(N)]
        )
        self.modelt.insert(0, self.linin)
        self.modelt.append(self.linout)
        self.model = nn.Sequential(*self.modelt)
        self.set_param()

    def set_param(self):
        """
        Sets parameters for all SSM layers in the deep model.
        """
        for i in range(1, len(self.modelt) - 1):
            if isinstance(
                self.modelt[i], SSM
            ):  # Check if it's an instance of CustomModule
                self.modelt[i].set_paramS()  # Call the custom method

    def forward(self, input):
        """
        Forward pass of the DeepLRU model.

        Args:
            input (tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            tensor: Output tensor of shape (batch_size, sequence_length, output_size).
        """
        result = self.model(input)
        return result
