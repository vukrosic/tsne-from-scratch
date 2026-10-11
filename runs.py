"""Every map of the video, saved in files/maps.json, so that the steps only read them.

`python3 runs.py` makes the maps that are missing (about 2 minutes for all of them)."""
import json
import os
import numpy as np
from tsne import (pictures, data_shares, make_map, keep_distances, two_groups, noise_pictures,
                  map_shares, gap, COUNT)

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'files', 'maps.json')
SEEDS = [0, 1, 2, 3, 4]
COUNTS = [2, 5, 30, 100, 300]
X, DIGITS, ORDER = pictures()
GROUPS, GROUP_OF = two_groups(X, DIGITS)
NOISE = noise_pictures()
_shares = {}


def shares(name, Z, count=COUNT):
    """The table of data shares for one set of pictures. Made once in a run of the program."""
    if (name, count) not in _shares:
        _shares[(name, count)] = data_shares(Z, count)
    return _shares[(name, count)]


def recipes():
    """name of a map -> a function that makes it and returns (dots, gap)."""
    r = {'keep': lambda: keep_distances(X)[0:2],
         'bell': lambda: make_map(shares('digits', X), tail=False)[0:2],
         'sizes': lambda: make_map(shares('groups', GROUPS))[0:2]}
    for s in SEEDS:
        r['seed%d' % s] = lambda s=s: make_map(shares('digits', X), seed=s)[0:2]
    for c in COUNTS:
        if c != COUNT:
            r['count%d' % c] = lambda c=c: make_map(shares('digits', X, c))[0:2]
    for c in (2, 30):
        r['noise%d' % c] = lambda c=c: make_map(shares('noise', NOISE, c))[0:2]
    return r


def load():
    if not os.path.exists(PATH):
        return {}
    with open(PATH) as f:
        return json.load(f)


def get(name):
    """The dots of one map (600 rows of 2 numbers) and its gap. For 'keep' the second number
    is the mean error of the distances in percent."""
    saved = load()
    if name not in saved:
        Y, g = recipes()[name]()
        saved = load()
        saved[name] = {'gap': round(float(g), 4), 'dots': [[round(float(v), 3) for v in row] for row in Y]}
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, 'w') as f:
            json.dump(saved, f, separators=(',', ':'))
    return np.array(saved[name]['dots']), saved[name]['gap']


if __name__ == '__main__':
    for name in recipes():
        Y, g = get(name)
        print('%-9s %8.4f   page %5.1f wide, %5.1f high' % (name, g, *(Y.max(0) - Y.min(0))))
