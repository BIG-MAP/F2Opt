"""Tools for working with compositions.

Resources:
    https://en.wikipedia.org/wiki/Compositional_data
    https://link.springer.com/book/10.1007/978-3-319-96422-5
    https://composition-stats.readthedocs.io/en/latest/
"""

import torch


# Basic operations


def closure(c: torch.Tensor, k=1.0) -> torch.Tensor:
    """Closure operator for compositions.

    Scale compositions c so that they sum to k.
    """
    c = torch.atleast_2d(c)
    assert c.ndim == 2
    c = c / c.sum(dim=1, keepdim=True)
    return k * c


def perturb(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Perturbation operator for compositions."""
    assert x.shape == y.shape
    x, y = closure(x), closure(y)
    return closure(x * y)


def perturb_inv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inverse perturbation operator for compositions."""
    assert x.shape == y.shape
    x, y = closure(x), closure(y)
    return closure(x / y)


def power(x: torch.Tensor, a: float) -> torch.Tensor:
    """Power operator for compositions."""
    x = closure(x)
    return closure(x ** a)


# Aitchison geometry


def inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Aitchison inner product for compositions."""
    x, y = closure(x), closure(y)
    a, b = clr(x), clr(y)
    return a @ b.T


def norm(x: torch.Tensor) -> torch.Tensor:
    """Aitchison norm for compositions."""
    return torch.sqrt(inner(x, x))


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Aitchison distance for compositions."""
    x, y = closure(x), closure(y)
    a, b = clr(x), clr(y)
    return torch.linalg.norm(a - b, dim=1)


# Transformations


def clr(x: torch.Tensor) -> torch.Tensor:
    """Centered log-ratio transformation."""
    lx = torch.log(closure(x))
    lgm = lx.mean(dim=1, keepdim=True)  # log geometric mean
    return lx - lgm


def clr_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse centered log-ratio transformation.

    Transform clr representation back to the original composition up to a scaling factor.
    """
    return closure(torch.exp(x))


def ilr(x: torch.Tensor) -> torch.Tensor:
    """Isometric log-ratio transformation."""
    basis = _pivot_basis(x.shape[-1])
    return inner(x, basis)


def ilr_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse isometric log-ratio transformation.

    Transform ilr representation back to the original composition up to a scaling factor.
    """
    basis = _pivot_basis(x.shape[-1] + 1)
    return clr_inv(x @ basis)  # TODO: basis.T?


def _pivot_basis(D: int) -> torch.Tensor:
    """Orthonormal basis for the ilr coordinates.

    Args:
        D: Dimension of the Aitchison simplex."""
    # basis = np.zeros((n, n-1))
    # for j in range(n-1):
    #     i = j + 1
    #     e = np.array([(1/i)]*i + [-1] +
    #                  [0]*(n-i-1))*np.sqrt(i/(i+1))
    #     basis[:, j] = e
    # return basis.T
    raise NotImplementedError


# Sampling


def sample_compositions(num_features: int, num_samples: int = 1) -> torch.Tensor:
    """Sample compositions.

    The sample space is the (num_features-1)-standard simplex.
    The values of each sampled composition are between 0 and 1 and all values sum to 1.

    Args:
        num_features: Number of features.
        num_samples: Number of samples.
    Returns:
        Tensor of sampled compositions (num_samples, num_features).
    """
    if num_features == 1:
        # If there is only one feature, the only possible composition is 1.0.
        return torch.ones(num_samples, 1)
    x = torch.rand(num_samples, num_features + 1)
    x[:, 0] = 0
    x[:, -1] = 1
    x, _ = torch.sort(x)  # cumulative compositions
    x = x[:, 1:] - x[:, :-1]  # compositions
    return x


def sample_compositions_with_constraints(
        lower: list[float], upper: list[float], tolerance: list[float],
        num_samples: int = 1, batch_size: int = 100) -> torch.Tensor:
    """Sample compositions with the given constraints.

    The sample space is the (num_features-1)-standard simplex with the given constraints.
    The values of each sampled composition are between lower and higher and all values sum to 1.

    Warning:
    While tolerance and lower bounds come for free, sampling is inefficient for tight upper bounds.
    Especially if the sum of the upper bounds is close to 1.0 and the number of features is large.

    Args:
        lower: Lower bounds (num_features,).
        upper: Upper bounds (num_features,).
        tolerance: Values below tolerance are rounded to zero (num_features,).
        num_samples: Number of samples.
        batch_size: Batch size for sampling.
    Returns:
        Tensor of sampled compositions (num_samples, num_features).
    """
    num_features = len(lower)
    lower, upper, tolerance = torch.tensor(lower), torch.tensor(upper), torch.tensor(tolerance)
    if num_features == 1:
        # If there is only one feature, the only possible composition is 1.0.
        assert len(lower) == len(upper) == len(tolerance) == 1
        assert upper.item() == 1.0
        return torch.ones(num_samples, 1)
    assert num_features >= 2, "At least two features are required."
    assert len(lower) == len(upper) == len(tolerance), \
        "Constraints must all have length (num_features,)."
    assert min(lower) >= 0 and max(lower) <= 1, "Lower bounds must be between 0 and 1."
    assert min(upper) >= 0 and max(upper) <= 1, "Upper bounds must be between 0 and 1."
    assert min(tolerance) >= 0 and max(tolerance) <= 1.0 / num_features, \
        "Tolerance must be between 0 and 1/num_features"
    assert ((lower == 0) | (lower >= tolerance)).all(), \
        f"Lower bounds must be zero or greater than tolerance {lower}."
    assert all(upper - lower >= 0.1), "The sample space is too constrained."
    assert sum(lower) <= 1.0, "The sample space is too constrained."
    assert sum(upper) >= 1.1, "The sample space is too constrained."
    # Sample compositions iteratively
    samples = []
    while sum([len(x) for x in samples]) < num_samples:
        # Sample unconstrained batch of candidate compositions
        candidates = sample_compositions(num_features, batch_size)
        # Apply tolerance
        if tolerance.sum() > 0:
            # Scale tolerance since lower bounds are added below
            mask = candidates < (tolerance - lower) / (1.0 - lower.sum())
            candidates[mask] = 0  # Set values below tolerance to zero
            candidates = candidates / candidates.sum(dim=1, keepdim=True)  # renormalize
        # Apply lower bounds
        candidates = lower + (1.0 - lower.sum()) * candidates
        # Apply upper bounds
        if upper.sum() < num_features:
            mask = (candidates <= upper).all(dim=1)
            candidates = candidates[mask]
        # Append to samples
        samples.append(candidates)
    samples = torch.cat(samples)
    return samples[:num_samples]
