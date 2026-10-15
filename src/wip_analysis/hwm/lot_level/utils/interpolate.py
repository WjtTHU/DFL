from datetime import datetime

from loguru import logger


def interpolate_between(
    start_time: datetime,
    end_time: datetime,
    start_idx: int,
    end_idx: int,
    weights: list[float],
) -> dict[int, datetime]:
    """
    Given start_idx at start_time, end_idx at end_time,
    interpolate a datetime for each indices between start_idx and end_idx

    Args:
        start_time (datetime): datetime at which start_idx takes place
        end_time (datetime): datetime at which end_idx takes place
        start_idx (int): index of start point
        end_idx (int): index of end point
        weights (list[float]): a weight list indicating a relative weight for each index

    Returns:
        dict[int, datetime]: indices in (start_idx, end_idx) (exclusive) each gets a datetime assigned
    """
    if end_time <= start_time or end_idx - start_idx <= 1:
        logger.warning(
            f"Invalid range: start_idx: '{start_idx:4d}', start_time: '{start_time}'"
            + f"               end_idx:   '{end_idx:4d}' end_time: '{end_time}'",
        )
        return {}

    if start_idx < 0:
        segment_weights = weights[0:end_idx]
        offset = 0
        # Include index 0 explicitly since it's not covered by the loop
        include_start = True
    else:
        segment_weights = weights[start_idx:end_idx]
        offset = start_idx
        include_start = False

    denom = sum(segment_weights)
    total_dur = end_time - start_time

    acc = 0.0
    res: dict[int, datetime] = {}
    if include_start:
        res[0] = start_time
    for local_i in range(len(segment_weights) - 1):
        acc += segment_weights[local_i]
        ratio = acc / denom
        res[offset + local_i + 1] = start_time + total_dur * ratio
    return res
