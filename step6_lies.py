"""Step 6. Three things the map shows that are not in the pictures."""
import numpy as np
from tsne import distances2, data_shares, spread, same_digit, neighbours_kept, nearest, COUNT
from runs import X, DIGITS, GROUPS, GROUP_OF, NOISE, SEEDS, COUNTS, get
from draw import text_map


def middles(Z, digits):
    return np.array([Z[digits == k].mean(0) for k in range(10)])


def nearest_group(Z, digits):
    """For every digit: the digit whose group middle is nearest."""
    return nearest(distances2(middles(Z, digits)), 1)[:, 0]


def clumping(Y):
    """Mean distance to the nearest dot, divided by the mean distance to the 10th nearest dot."""
    d = np.sort(np.sqrt(distances2(Y)), axis=1)
    return float(d[:, 1].mean() / d[:, 10].mean())


def lie_sizes():
    print('1. THE SIZE OF AN ISLAND')
    a, b = GROUP_OF == 0, GROUP_OF == 1
    print('   made-up pictures: 300 copies of one 3 with small changes, 300 copies of one 7 with changes 10 times as big')
    print('   in the 64 numbers: the 3 group is %.3f wide, the 7 group %.3f wide: %.1f times as wide' % (spread(GROUPS, a), spread(GROUPS, b), spread(GROUPS, b) / spread(GROUPS, a)))
    Y, _ = get('sizes')
    print('   on the map:        the 3 group is %.2f wide, the 7 group %.2f wide: %.2f times as wide' % (spread(Y, a), spread(Y, b), spread(Y, b) / spread(Y, a)))
    _, widths = data_shares(GROUPS, give_widths=True)
    print('   why: every bell is set to %d neighbours. Mean width of a bell: %.3f in the 3 group, %.3f in the 7 group' % (COUNT, widths[a].mean(), widths[b].mean()))
    print('   the real digits: the zeros are %.2f wide in the 64 numbers and the ones %.2f (%.1f times). On the map: %.1f and %.1f (%.1f times)' % (
        spread(X, DIGITS == 0), spread(X, DIGITS == 1), spread(X, DIGITS == 1) / spread(X, DIGITS == 0),
        spread(get('seed0')[0], DIGITS == 0), spread(get('seed0')[0], DIGITS == 1),
        spread(get('seed0')[0], DIGITS == 1) / spread(get('seed0')[0], DIGITS == 0)))


def lie_gaps():
    print('\n2. WHICH ISLAND LIES NEXT TO WHICH')
    truth = nearest_group(X, DIGITS)
    print('   the same pictures, the same program, five different random clouds at the start:')
    print('   start   gap     same digit   pictures kept   island nearest to the island of 0 1 2 3 4 5 6 7 8 9')
    maps = [get('seed%d' % s) for s in SEEDS]
    for s, (Y, g) in zip(SEEDS, maps):
        ng = nearest_group(Y, DIGITS)
        print('   %d      %.3f     %.1f %%        %.1f %%                                        %s   (%d of 10 as in the pictures)' % (
            s, g, 100 * same_digit(Y, DIGITS), 100 * neighbours_kept(X, Y), ' '.join(str(v) for v in ng), (ng == truth).sum()))
    print('   in the 64 numbers, the group nearest to the group of 0 1 2 3 4 5 6 7 8 9:         %s' % ' '.join(str(v) for v in truth))
    a, b = nearest(distances2(maps[0][0]), 10), nearest(distances2(maps[1][0]), 10)
    stay = np.mean([len(set(a[i]) & set(b[i])) / 10.0 for i in range(len(a))])
    print('   the near dots stay: of the 10 nearest dots of a dot on map 0, %.1f %% are among its 10 nearest on map 1' % (100 * stay))


def lie_count():
    print('\n3. THE NEIGHBOUR COUNT IS YOUR CHOICE')
    print('   count   gap     same digit   pictures kept   the page is')
    for c in COUNTS:
        Y, g = get('seed0' if c == COUNT else 'count%d' % c)
        print('   %3d    %.3f     %.1f %%        %.1f %%         %5.1f wide' % (c, g, 100 * same_digit(Y, DIGITS), 100 * neighbours_kept(X, Y), np.ptp(Y[:, 0])))
    print('\n   count 2:\n' + text_map(get('count2')[0], DIGITS, rows=20))
    print('\n   count 300:\n' + text_map(get('count300')[0], DIGITS, rows=20))
    print('\n   600 pictures of noise: every square is a random number. There are no groups.')
    d = np.sqrt(distances2(NOISE))
    np.fill_diagonal(d, np.inf)
    print('   in the 64 numbers: the nearest picture is %.2f away, the farthest %.2f' % (d.min(1).mean(), np.where(np.isinf(d), 0, d).max(1).mean()))
    flat = np.random.RandomState(0).rand(600, 2)
    print('   nearest dot / 10th nearest dot. 600 dots thrown on a page at random: %.2f' % clumping(flat))
    for c in (COUNT, 2):
        Y, g = get('noise%d' % c)
        print('   map of the noise, count %2d: %.2f' % (c, clumping(Y)))
    print('\n   noise, count 2:\n' + text_map(get('noise2')[0], rows=20))


if __name__ == '__main__':
    lie_sizes()
    lie_gaps()
    lie_count()
