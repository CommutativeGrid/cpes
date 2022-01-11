import numpy as np

from cpes.aux_functions import *


def test_distance_array():
    data = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    d = distance_array(data)
    assert d[1] == 1
    assert np.round(d[2], 3) == 1.414
