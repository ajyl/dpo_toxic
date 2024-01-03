"""
Utility functions.
"""

import os

VERBOSE = os.getenv("VERBOSE", 0)


def verbose_print(x):
    if VERBOSE:
        print(x)
