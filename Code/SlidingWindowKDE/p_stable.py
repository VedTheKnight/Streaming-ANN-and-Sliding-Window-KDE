import torch,random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PStableLSH:
    def __init__(self, dim, w, p=2, seed=None):
        """
        dim : int  -> dimensionality of input vectors
        w   : float -> bucket width
        p   : float -> p-stable distribution (p=1 for Cauchy, p=2 for Gaussian)
        seed : int  -> optional random seed
        """
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

        self.dim = dim
        self.w = w
        self.p = p
        if p == 2:
            # Gaussian (for L2)
            self.a = torch.randn(dim, generator=self.rng, device=device)
        elif p == 1:
            # Cauchy (for L1)
            self.a = torch.distributions.Cauchy(0.0, 1.0).sample((dim,)).to(device)
        else:
            raise ValueError("Only p=1 (Cauchy) or p=2 (Gaussian) supported here.")
        self.b = torch.empty(1, device=device).uniform_(0, w).item()

    def hash(self, v):
        """
        v : torch.Tensor -> input vector (dim,)
        Returns: integer bucket index (can be negative)
        """
        projection = torch.dot(self.a, v)
        return int(torch.floor((projection + self.b) / self.w).item())


class UniversalHasher:
    def __init__(self, R, seed=None):
        if seed is not None:
            random.seed(seed)
        self.R = R
        self.A = np.uint64(random.getrandbits(64) | 1)
        self.B = np.uint64(random.getrandbits(64))

    def hash(self, x):
        x64 = np.array(x).astype(np.uint64)
        h = (self.A * x64 + self.B) & np.uint64((1 << 64) - 1)
        return int(h % self.R) + 1