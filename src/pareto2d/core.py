import enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    O1_MAX_O2_MAX = 1
    O1_MAX_O2_MIN = 2
    O1_MIN_O2_MAX = 3
    O1_MIN_O2_MIN = 4


def _has_duplicates(o1, o2):
    ii = np.lexsort((o2, o1))
    o1_sorted = o1[ii]
    o2_sorted = o2[ii]
    delta_o1 = o1_sorted[1:] - o1_sorted[:-1]
    delta_o2 = o2_sorted[1:] - o2_sorted[:-1]
    n = np.sum((delta_o1 == 0) & (delta_o2 == 0))
    return n > 0


def get_dedpuplicated_copy(o1, o2):
    ii = np.lexsort((o2, o1))
    o1_sorted = o1[ii]
    o2_sorted = o2[ii]
    delta_o1 = o1_sorted[1:] - o1_sorted[:-1]
    delta_o2 = o2_sorted[1:] - o2_sorted[:-1]
    non_duplicates = (delta_o1 != 0) | (delta_o2 != 0)
    nn = len(o1) - np.sum(non_duplicates) - 1
    logger.info(f"Removing {nn} duplicates")
    non_duplicates = non_duplicates.tolist()
    non_duplicates.append(True)
    non_duplicates = np.array(non_duplicates)
    return o1[ii[non_duplicates]], o2[ii[non_duplicates]]


def _get_pf_mask_max_max(o1, o2):
    # o1, o2 -> maximized
    # assert not _has_duplicates(o1, o2)
    o1_, o2_ = get_dedpuplicated_copy(o1, o2)
    n = len(o1_)
    pareto_pairs = set()

    for i in range(n):
        o1i = o1_[i]
        o2i = o2_[i]
        is_pareto = np.sum((o1_ >= o1i) & (o2_ >= o2i)) <= 1
        if is_pareto:
            pareto_pairs.add((o1i, o2i))

    return np.fromiter((pair in pareto_pairs for pair in zip(o1, o2)), dtype=bool)


def get_pf_mask(o1, o2, mode):
    if mode == Mode.O1_MAX_O2_MAX:
        return _get_pf_mask_max_max(o1, o2)
    elif mode == Mode.O1_MAX_O2_MIN:
        return _get_pf_mask_max_max(o1, -o2)
    elif mode == Mode.O1_MIN_O2_MAX:
        return _get_pf_mask_max_max(-o1, o2)
    elif mode == Mode.O1_MIN_O2_MIN:
        return _get_pf_mask_max_max(-o1, -o2)
    else:
        raise ValueError(f"Unknown {mode=}")


def _get_sorted_pf_max_max(o1, o2):
    assert not _has_duplicates(o1, o2)
    mask = _get_pf_mask_max_max(o1, o2)
    o1_pf = o1[mask]
    o2_pf = o2[mask]
    ii = np.lexsort((o2_pf, o1_pf))
    return o1_pf[ii], o2_pf[ii]


def get_sorted_pf(o1, o2, mode):
    if mode == Mode.O1_MAX_O2_MAX:
        return _get_sorted_pf_max_max(o1, o2)
    elif mode == Mode.O1_MAX_O2_MIN:
        o1_pf, o2_pf = _get_sorted_pf_max_max(o1, -o2)
        return o1_pf, -o2_pf
    elif mode == Mode.O1_MIN_O2_MAX:
        o1_pf, o2_pf = _get_sorted_pf_max_max(-o1, o2)
        return -np.flip(o1_pf), np.flip(o2_pf)
    elif mode == Mode.O1_MIN_O2_MIN:
        o1_pf, o2_pf = _get_sorted_pf_max_max(-o1, -o2)
        return -np.flip(o1_pf), -np.flip(o2_pf)
    else:
        raise ValueError(f"Unknown {mode=}")


def _get_pf_o2_max_max_one_value(o1, o1_pf_sorted, o2_pf_sorted):
    i = 0
    found = False
    n = len(o1_pf_sorted)
    while i < n:
        if o1_pf_sorted[i] > o1:
            found = True
            break
        i += 1
    if found:
        return o2_pf_sorted[i]
    else:
        return 0.0


def _get_pf_o2_max_max(o1, o1_pf, o2_pf):
    ii = np.lexsort((o2_pf, o1_pf))
    o1_pf_sorted = o1_pf[ii]
    o2_pf_sorted = o2_pf[ii]
    return np.fromiter(
        (_get_pf_o2_max_max_one_value(o1i, o1_pf_sorted, o2_pf_sorted) for o1i in o1),
        o1.dtype,
    )


def interpolate_pf_o2(o1, o1_pf_sorted, o2_pf_sorted, mode):
    if mode == Mode.O1_MAX_O2_MAX:
        return _get_pf_o2_max_max(o1, o1_pf_sorted, o2_pf_sorted)
    elif mode == Mode.O1_MAX_O2_MIN:
        return -_get_pf_o2_max_max(o1, o1_pf_sorted, -o2_pf_sorted)
    elif mode == Mode.O1_MIN_O2_MAX:
        return _get_pf_o2_max_max(-o1, -o1_pf_sorted, o2_pf_sorted)
    elif mode == Mode.O1_MIN_O2_MIN:
        return -_get_pf_o2_max_max(-o1, -o1_pf_sorted, o2_pf_sorted)
    else:
        raise ValueError(f"Unknown {mode=}")


def interpolate_pf_o1(o2, o1_pf_sorted, o2_pf_sorted, mode):
    if mode == Mode.O1_MAX_O2_MAX or mode == Mode.O1_MIN_O2_MIN:
        return interpolate_pf_o2(o2, o2_pf_sorted, o1_pf_sorted, mode=mode)
    elif mode == Mode.O1_MAX_O2_MIN:
        return interpolate_pf_o2(o2, o2_pf_sorted, o1_pf_sorted, mode=Mode.O1_MIN_O2_MAX)
    elif mode == Mode.O1_MIN_O2_MAX:
        return interpolate_pf_o2(o2, o2_pf_sorted, o1_pf_sorted, mode=Mode.O1_MAX_O2_MIN)
    else:
        raise ValueError(f"Unknown {mode=}")
