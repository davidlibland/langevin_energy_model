def add_soft_constraint(
    loss: float,
    constrained_value: float,
    lower_bound: float = None,
    upper_bound: float = None,
    epsilon=0.25,
) -> float:
    """Adds a soft constraint."""
    if lower_bound is not None:
        if constrained_value <= lower_bound:
            return float("inf")
        if constrained_value < lower_bound + epsilon:
            soft_penalty = 1 / (constrained_value - lower_bound) - 1 / epsilon
            loss += soft_penalty
    if upper_bound is not None:
        if constrained_value >= upper_bound:
            return float("inf")
        if constrained_value > upper_bound - epsilon:
            soft_penalty = 1 / (upper_bound - constrained_value) - 1 / epsilon
            loss += soft_penalty
    return loss
