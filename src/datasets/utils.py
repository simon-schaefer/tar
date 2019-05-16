#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Utility functions for dataset functions.
# =============================================================================
import sys

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int:
    """ Draws progress bar showing the number of executed
    iterations over the overall number of iterations.
    Increments the iteration and returns it. """
    status = ""
    progress = float(iteration) / float(num_steps)
    if progress >= 1.0:
        progress, status = 1.0, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (bar_length - block), round(progress*100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return iteration + 1
