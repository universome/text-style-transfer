def compute_param_by_scheme(scheme, num_iters_done):
    """
    :param scheme: format (start_val, end_val, period)
    """
    t1, t2, period = scheme

    if num_iters_done > period:
        return t2
    else:
        return t1 - (t1 - t2) * num_iters_done / period
