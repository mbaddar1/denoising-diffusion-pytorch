"""
This script is to test the sinkhorn implementation of sinkhorn
here
https://github.com/dfdazac/wassdistance/blob/c145dd9e92d29a05e6747beb8914210ad8570b3d/layers.py#L5
illustrated here
https://github.com/dfdazac/wassdistance/blob/master/sinkhorn.ipynb
with the set of code snippets and toy datasets provided
"""
import numpy as np
import torch

from stat_dist.layers import SinkhornDistance

if __name__ == '__main__':
    n_points = 5
    a = np.array([[i, 0] for i in range(n_points)])
    b = np.array([[i, 1] for i in range(n_points)])
    device = torch.device("cuda")
    x = torch.tensor(a, dtype=torch.float, device=device)
    y = torch.tensor(b, dtype=torch.float, device=device)
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=device)
    dist, P, C = sinkhorn(x, y)
    print("Sinkhorn distance: {:.3f}".format(dist.item()))
