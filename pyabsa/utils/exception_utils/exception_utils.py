# -*- coding: utf-8 -*-
# file: exception_utils.py
# time: 02/11/2022 17:15
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import functools
import multiprocessing


def time_out(max_timeout):
    """Timeout decorator, parameter in seconds."""

    def timeout_decorator(item):
        """Wrap the original function."""

        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            try:
                pool = multiprocessing.Pool(processes=1)
                async_result = pool.apply_async(item, args=args, kwds=kwargs)
                return async_result.get(max_timeout)
            except Exception as e:
                return None

        return func_wrapper

    return timeout_decorator
