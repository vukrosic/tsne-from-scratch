"""The exercise. Three made-up groups of 200 rows of 64 numbers each, on one line:
group B is 1 away from group A, group C is 10 away from group A.

    python3 exercise.py          neighbour count 30, as in the video
    python3 exercise.py 300      another neighbour count (2 to 590)

Guess first: on the map, how many times as far from A is C as B is? Then run it.
"""
import sys
import numpy as np
from tsne import data_shares, make_map, spread
from draw import text_map

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def three_groups(far=10.0, noise=0.03, each=200, seed=3):
    """Group middles at 0, 1 and `far` in the first number. Every number gets a small random change."""
    rs = np.random.RandomState(seed)
    Z = rs.randn(3 * each, 64) * noise
    Z[each:2 * each, 0] += 1.0
    Z[2 * each:, 0] += far
    return Z, np.repeat([0, 1, 2], each)


def between(Z, group, a, b):
    return float(np.sqrt(((Z[group == a].mean(0) - Z[group == b].mean(0)) ** 2).sum()))


if __name__ == '__main__':
    Z, group = three_groups()
    ab, ac = between(Z, group, 0, 1), between(Z, group, 0, 2)
    print('in the 64 numbers: A to B %.2f, A to C %.2f: C is %.1f times as far. Every group is %.2f wide' % (ab, ac, ac / ab, spread(Z, group == 0)))
    for seed in (0, 1):
        Y, g, _ = make_map(data_shares(Z, COUNT), seed=seed)
        ab, ac = between(Y, group, 0, 1), between(Y, group, 0, 2)
        print('map with neighbour count %d, start %d: A to B %.1f, A to C %.1f: C is %.2f times as far (gap %.3f)' % (COUNT, seed, ab, ac, ac / ab, g))
    print('\n' + text_map(Y, np.array(list('ABC'))[group], rows=18))
