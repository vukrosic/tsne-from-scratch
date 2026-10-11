"""t-SNE from an empty file: pictures, distances, shares, a flat map, pulls.

Words of the video -> words of the paper (van der Maaten and Hinton, 2008):
  picture         datapoint x_i (here 64 numbers)
  dot             map point y_i (2 numbers: across and up)
  share           p_j|i, p_ij (data) and q_ij (map)
  width           sigma_i of the bell around picture i
  neighbour count perplexity
  gap             Kullback-Leibler divergence, the cost C
  pull            minus the gradient of the gap for one dot
"""
import numpy as np
from sklearn.datasets import load_digits

N = 600            # pictures on the map
COUNT = 30         # neighbour count (perplexity) used in the video
STEPS = 1000       # map steps, as in the paper
BOOST = 4.0        # data shares are multiplied by this ...
BOOST_STEPS = 50   # ... during the first steps ("early exaggeration")
RATE = 100.0       # step size (learning rate)


def pictures(n=N, seed=0):
    """n pictures of the digits set in a fixed random order: 64 numbers from 0 to 1 each."""
    d = load_digits()
    order = np.random.RandomState(seed).permutation(len(d.data))[:n]
    return d.data[order] / 16.0, d.target[order], order


def distances2(X):
    """Squared distance between every two rows: the sum of the squared differences."""
    s = (X * X).sum(1)
    with np.errstate(all='ignore'):   # numpy on a Mac warns about this product; the numbers are right
        d2 = s[:, None] + s[None, :] - 2.0 * X @ X.T
    np.fill_diagonal(d2, 0.0)
    return np.maximum(d2, 0.0)


def shares_from(d2_row, i, width):
    """Shares that picture i gives to every other picture: a bell of this width over the distance."""
    near = d2_row[np.arange(len(d2_row)) != i].min()   # taken off first, so that the nearest picture never rounds to 0
    w = np.exp(-np.maximum(d2_row - near, 0.0) / (2.0 * width * width))
    w[i] = 0.0
    return w / w.sum()


def neighbour_count(p):
    """2 to the power of the entropy. 30 equal shares give 30, one share of 100 % gives 1."""
    q = p[p > 1e-300]
    return float(2.0 ** (-(q * np.log2(q)).sum()))


def find_width(d2_row, i, count=COUNT, rounds=60):
    """Halve the search range until the shares of picture i have this neighbour count."""
    lo, hi = 1e-4, 100.0
    for _ in range(rounds):
        mid = (lo + hi) / 2.0
        if neighbour_count(shares_from(d2_row, i, mid)) > count:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def data_shares(X, count=COUNT, give_widths=False):
    """One table of shares for all pairs of pictures. The whole table adds up to 1."""
    d2 = distances2(X)
    n = len(X)
    P = np.zeros((n, n))
    widths = np.zeros(n)
    for i in range(n):
        widths[i] = find_width(d2[i], i, count)
        P[i] = shares_from(d2[i], i, widths[i])
    P = (P + P.T) / (2.0 * n)
    return (P, widths) if give_widths else P


def map_weights(Y, tail=True):
    """Weight of every pair of dots. tail=True: 1 / (1 + distance squared). False: a bell."""
    d2 = distances2(Y)
    w = 1.0 / (1.0 + d2) if tail else np.exp(-d2)
    np.fill_diagonal(w, 0.0)
    return w


def map_shares(Y, tail=True):
    w = map_weights(Y, tail)
    return w / w.sum()


def gap(P, Q):
    """0 when the two tables are equal. It grows when near pictures are drawn far apart."""
    m = P > 1e-300
    return float((P[m] * np.log(P[m] / np.maximum(Q[m], 1e-300))).sum())


def pulls(P, Y, tail=True):
    """The total pull on every dot. A pair with a data share above its map share pulls together."""
    w = map_weights(Y, tail)
    Q = w / w.sum()
    S = (P - Q) * w if tail else (P - Q)
    with np.errstate(all='ignore'):
        return -4.0 * (S.sum(1)[:, None] * Y - S @ Y), Q


def make_map(P, steps=STEPS, seed=0, tail=True, boost=BOOST, boost_steps=BOOST_STEPS,
             rate=RATE, keep=()):
    """Start from a small random cloud and move every dot along its pull, `steps` times.

    Returns the dots, the gap at the end, and the dots at the steps listed in `keep`."""
    n = len(P)
    rs = np.random.RandomState(seed)
    Y = rs.randn(n, 2) * 1e-2
    last = np.zeros_like(Y)
    gains = np.ones_like(Y)
    kept = {}
    if 0 in keep:
        kept[0] = Y.copy()
    for t in range(1, steps + 1):
        F, Q = pulls(P * boost if t <= boost_steps else P, Y, tail)
        # a dot that keeps its direction takes bigger steps (Jacobs, 1988)
        same = (F > 0) == (last > 0)
        gains = np.maximum(np.where(same, gains + 0.2, gains * 0.8), 0.01)
        last = (0.5 if t < 250 else 0.8) * last + rate * gains * F
        Y = Y + last
        Y = Y - Y.mean(0)
        if t in keep:
            kept[t] = Y.copy()
    return Y, gap(P, map_shares(Y, tail)), kept


def keep_distances(X, steps=600, seed=0):
    """The simple try: move the dots until every distance on the page is as close as possible
    to the distance between the two pictures. Returns the dots and the mean error in percent."""
    n = len(X)
    D = np.sqrt(distances2(X))
    Y = np.random.RandomState(seed).randn(n, 2) * 1e-2
    for _ in range(steps):
        d = np.sqrt(distances2(Y)) + np.eye(n)
        S = (d - D) / d                      # above 0: this pair is drawn too far apart
        np.fill_diagonal(S, 0.0)
        with np.errstate(all='ignore'):
            Y = Y - (S.sum(1)[:, None] * Y - S @ Y) / n
    d = np.sqrt(distances2(Y))
    return Y, float(100.0 * np.abs(d - D).sum() / D.sum())


def two_groups(X, digits, small=0.02, big=0.2, each=300, seed=1):
    """Made-up pictures: `each` copies of one 3 with small changes, `each` copies of one 7 with big ones."""
    rs = np.random.RandomState(seed)
    a = X[np.where(digits == 3)[0][0]] + rs.randn(each, 64) * small
    b = X[np.where(digits == 7)[0][0]] + rs.randn(each, 64) * big
    return np.clip(np.vstack([a, b]), 0.0, 1.0), np.array([0] * each + [1] * each)


def noise_pictures(n=N, seed=2):
    """Pictures with no groups at all: every square is a random number from 0 to 1."""
    return np.random.RandomState(seed).rand(n, 64)


def spread(Z, group):
    """How wide a group is: the mean distance of its members from the middle of the group."""
    return float(np.sqrt(((Z[group] - Z[group].mean(0)) ** 2).sum(1)).mean())


def nearest(d2, k):
    """For every row the k nearest other rows."""
    d = d2.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def same_digit(Y, digits, k=10):
    """Of the k nearest dots of a dot, the share that shows the same digit. Mean over all dots."""
    nn = nearest(distances2(Y), k)
    return float((digits[nn] == digits[:, None]).mean())


def neighbours_kept(X, Y, k=10):
    """Of the k nearest dots of a dot, the share that is also among its k nearest pictures."""
    a = nearest(distances2(X), k)
    b = nearest(distances2(Y), k)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))
