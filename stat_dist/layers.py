import logging

import torch
import torch.nn as nn
from tqdm import tqdm

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# GitHub repo https://github.com/dfdazac/wassdistance/tree/master
logger = logging.getLogger()


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, device, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device = device

    def forward(self, x, y):
        """
        Assumes data in one of two shapes
        x,y => N X D where N is the number of points, D is each point dimension
        x,y => B X N X D where B is the batch size
        It's preferable to start with N X D to make calculations tractable. D can be the flattened dim
        D1 X D2 X ... X D_M
        """
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        # adding extra dim check, based on the code line
        # https://github.com/dfdazac/wassdistance/blob/master/layers.py#L34
        if x.dim() == 2:
            logger.info(f"Dim = 2 and using direct sinkhorn calculations.\n")
            batch_size = 1
        elif x.dim() == 3:
            batch_size = x.shape[0]
            logger.info(f"Dim = 3 and using batched sinkhorn calculation with first dimension as the batch size.\n")
        else:
            raise ValueError(f"data dim must be 2 or 3 . x and y shape is B X N X D where B is batch size, "
                             f"N is number of points for each "
                             f"batch and D is the (flattened) dim for each point. "
                             f"Received data ndim = {x.dim()}")
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in tqdm(range(self.max_iter), desc="sinkhorn iterations"):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
