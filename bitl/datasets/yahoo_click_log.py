# coding=utf-8
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import gzip
import traceback

import numpy as np
from scipy.sparse import csr_matrix

NUM_FEATURES = 136

def load_clicklog(filepath, n_samples=None):
    """
    Parse Yahoo! Click log compressed archives

    Parameters
    ----------
    filepath : str
        The path to the compressed Click log archive

    n_samples : int, optional
        The number of samples to keep. (default: all)

    Returns
    -------
    stream_array: array of shape [n_samples, 4]
        Each sample is an array of:
        * article id
        * reward (click/no click),
        * sparse feature vector
        * timestamp of the observation

    items_counter: integer
        The number of articles contained in the stream.
        The article ids are unique integers but do not start at 0,
        so we convert them to be zero-based and monotonic.

    """
    stream_array = []

    # Normalize article IDs: Integer sequence starting at 0
    items_counter = 0
    items_index = dict()
    try:
        with gzip.open(filepath) as logfile:
            for _ in range(n_samples):
                splits = logfile.readline().decode().split()
                n_splits = len(splits)
                timestamp = int(splits[0])
                click = int(splits[2])
                article_id = int(splits[1].split('-')[1])
                if article_id in items_index:
                    article_id = items_index[article_id]
                else:
                    items_index[article_id] = items_counter
                    article_id = items_counter
                    items_counter += 1

                features = np.zeros(NUM_FEATURES, dtype=np.uint8)
                i = 5
                # Parsing feature vector
                while i < n_splits:
                    if splits[i][0] != '|':
                        features[int(splits[i]) - 1] = 1
                    i += 1
                sample = np.array([
                    article_id, click, csr_matrix(features), timestamp])
                stream_array.append(sample)
    except FileNotFoundError:
        traceback.print_exc()

    stream_array = np.asarray(stream_array, dtype=np.dtype('O'))

    return stream_array, items_counter
