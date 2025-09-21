import numpy as np

class PStableLSH:
    def __init__(self, dim, w, p=2, seed=None):
        """
        dim : int  -> dimensionality of input vectors
        w   : float -> bucket width
        p   : float -> p-stable distribution (p=1 for Cauchy, p=2 for Gaussian)
        seed : int  -> optional random seed
        """
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.w = w
        self.p = p

        if p == 2:
            # Gaussian (L2)
            self.a = self.rng.normal(0, 1, size=dim)
        elif p == 1:
            # Cauchy (L1)
            self.a = self.rng.standard_cauchy(size=dim)
        else:
            raise ValueError("Only p=1 (Cauchy) or p=2 (Gaussian) supported.")

        self.b = self.rng.uniform(0, w)

    def hash(self, v):
        """
        v : ndarray (..., dim) -> input vector or batch of vectors
        Returns: integer bucket index (or array of indices)
        """
        v = np.asarray(v, dtype=np.float32)
        projection = np.dot(v, self.a)  # works for (dim,) or (N,dim)
        return np.floor((projection + self.b) / self.w).astype(np.int64)


class UniversalHasher:
    def __init__(self, R, seed=None):
        rng = np.random.default_rng(seed)
        self.R = np.uint64(R)
        self.A = np.uint64(rng.integers(1, 1 << 63) | 1)  # odd to avoid trivial hash
        self.B = np.uint64(rng.integers(0, 1 << 63))

    def hash(self, x):
        """
        x : int or array-like of int -> input key(s)
        Returns: int or ndarray of hashed indices in range [1, R]
        """
        x64 = np.asarray(x, dtype=np.uint64)
        h = (self.A * x64 + self.B) & np.uint64((1 << 64) - 1)
        return (h % self.R) + 1
