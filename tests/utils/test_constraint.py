from src.utils.constraints import add_soft_constraint


def test_add_soft_constraint():
    """Tests that the soft constraint works."""
    loss = 35
    assert (
        add_soft_constraint(loss, 3) == loss
    ), "Unconstrained value should not be altered"

    assert (
        add_soft_constraint(loss, 3, lower_bound=1, upper_bound=4) == loss
    ), "Unconstrained value should not be altered"

    assert add_soft_constraint(loss, 0, lower_bound=1, upper_bound=4) == float(
        "inf"
    ), "invalid value should be inf"
    assert add_soft_constraint(loss, 0, lower_bound=1) == float(
        "inf"
    ), "invalid value should be inf"
    assert add_soft_constraint(loss, 5, lower_bound=1, upper_bound=4) == float(
        "inf"
    ), "invalid value should be inf"
    assert add_soft_constraint(loss, 5, upper_bound=4) == float(
        "inf"
    ), "invalid value should be inf"

    assert (
        add_soft_constraint(loss, 1.25, lower_bound=1, upper_bound=4, epsilon=0.5)
        > loss
    ), "near invalid value should be larger"
    assert (
        add_soft_constraint(loss, 1.25, lower_bound=1, epsilon=0.5) > loss
    ), "near invalid value should be larger"
    assert (
        add_soft_constraint(loss, 3.75, lower_bound=1, upper_bound=4, epsilon=0.5)
        > loss
    ), "near invalid value should be larger"
    assert (
        add_soft_constraint(loss, 3.75, upper_bound=4, epsilon=0.5) > loss
    ), "near invalid value should be larger"
