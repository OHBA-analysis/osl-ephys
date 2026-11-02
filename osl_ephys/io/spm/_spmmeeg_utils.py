import numpy as np


def _empty_to_val(var, new_val):
    return new_val if isinstance(var, np.ndarray) and var.size == 0 else var


def empty_to_none(var):
    return _empty_to_val(var, None)


def empty_to_zero(var):
    return _empty_to_val(var, 0)


def check_lowered_string(array, search_term):
    return np.char.find(np.char.lower(array), search_term.lower()) != -1
