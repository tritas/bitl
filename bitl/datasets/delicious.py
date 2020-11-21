# coding=utf-8
# Author: Aris Tritas
# License: BSD 3-clause
import traceback

import numpy as np


def load_delicious(filepath):
    """
    TODO: Encode the tag counts as normalized vectors over all tags

    An example entry:
    http://boingboing.net/	11053	2002-11-15
    blog	5018
    news	2763
    ... (8 more)

    The stream is the a (URL, visits, date, tags) tuple

    Reference
    ---------
    R5 - Yahoo! Delicious Popular URLs and Tags, version 1.0
    https://webscope.sandbox.yahoo.com/catalog.php?datatype=r

    """
    stream_array = []
    try:
        with open(filepath) as logfile:
            for ind, line in enumerate(logfile):
                splits = line.split()
                # Read the ten tags and their counts
                tag_counts = np.empty((2, 10), dtype=np.dtype("O"))
                for i, loc in enumerate(range(3, len(splits) - 1, 2)):
                    tag_counts[:, i] = [splits[loc], int(splits[loc + 1])]

                sample = np.array(
                    [splits[0], int(splits[1]), splits[2], tag_counts]
                )
                stream_array.append(sample)
    except FileNotFoundError:
        traceback.print_exc()
    stream_array = np.asarray(stream_array, dtype=np.dtype("O"))
    return stream_array
