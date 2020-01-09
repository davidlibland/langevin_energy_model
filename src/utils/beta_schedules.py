"""
Utilities to build beta schedules.
"""
import typing

import numpy as np


def build_schedule(
    *instructions: typing.Tuple[str, float, int], start=0.0
) -> np.ndarray:
    """
    Parses instructions of the form:
        ("geom", 0.1, 5), ("arith", 0.5, 10), ..., ("geom", 1, 6)
    To produce a geometric sequence of length 5 (from 0 upto .1) followed by
    an arithmetic sequence of length 10 (upto 0.5), ... followed by
    a geometric sequence of length 6 (upto 1)

    Note: Radford Neal suggests using a geometric schedule (but sometimes
    flattens it near zero to an arithmetic schedule). AIS is most
    computationally efficient when the schedule length is equal to the
    variance of the log-weights (which is one of the scalar metrics output
    by this Metric).
    (cf. https://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf)

    Note: The first instruction must be arithmetic (to hit zero), while the
    last instruction must end at 1.

    Args:
        instructions: The instructions used to build the schedule.
        start: An optional starting point, defaults to zero.

    Returns:
        np.ndarray (the schedule).
    """
    sequence = []
    stop = 0
    for seq_type, stop, num in instructions:
        assert 0 < stop <= 1, (
            f"Invalid input: {seq_type, stop, num} "
            f"stops must lie in the half open interval."
        )
        if seq_type.lower()[0] == "a":
            # Arithmetic case
            seq = np.linspace(start=start, stop=stop, num=num)
        elif seq_type.lower()[0] == "g":
            # Geometric case
            if start == 0:
                raise ValueError(
                    "Instructions must start with an " "arithmetic sequence"
                )
            seq = np.exp(np.linspace(start=np.log(start), stop=np.log(stop), num=num))
        else:
            raise ValueError(f"Unrecognized sequence type: {seq_type}")
        sequence.append(seq)
        start = stop
    assert stop == 1, "The instructions must end at 1."
    schedule = np.concatenate(sequence).flatten()
    schedule[-1] = 1  # Ensure it ends at 1 (despite numerical errors).
    return schedule
