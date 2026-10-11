"""Step 4. Dots on a page get shares too. Move the dots until the two tables agree."""
import numpy as np
from tsne import distances2, map_shares, gap, pulls, make_map, same_digit, neighbours_kept
from runs import X, DIGITS, shares, get
from draw import text_map
from step1_distances import A, SEVEN, near

B = int(near[A][0])                     # the nearest picture of picture A
KEEP = (0, 1, 10, 50, 100, 250, 1000)

if __name__ == '__main__':
    P = shares('digits', X)
    Y = np.random.RandomState(0).randn(len(X), 2) * 1e-2
    print('start: %d dots in a small random cloud, %.3f wide. Dot %d is at (%.4f, %.4f)' % (len(Y), np.ptp(Y[:, 0]), A, *Y[A]))
    print('\nshares on the page: the same bell, with one width for all dots, over the distance on the page')
    for d in (0.0, 0.5, 1.0, 2.0, 3.0):
        print('  two dots %.1f apart: weight %.4f' % (d, np.exp(-d * d)))
    print('  every weight is divided by the sum of all weights: the table of the page adds up to 100 % too')
    Q = map_shares(Y, tail=False)
    F, _ = pulls(P, Y, tail=False)
    print('\nthe two tables at the start (percent of the whole table):')
    for name, j in (('its nearest picture', B), ('the 7', SEVEN)):
        side = 'pulled together' if P[A, j] > Q[A, j] else 'pushed apart'
        print('  picture %d and %s (picture %d): pictures %.5f %%, page %.5f %% -> %s' % (A, name, j, 200 * P[A, j], 200 * Q[A, j], side))
    print('  the gap adds up, over all pairs: share of the pictures times log(share of the pictures / share of the page)')
    print('  gap at the start: %.3f. Equal tables give 0' % gap(P, Q))
    print('the pull between two dots: 4 times (share of the pictures minus share of the page) times their distance on the page')
    for name, j in (('its nearest picture', B), ('the 7', SEVEN)):
        dist = np.sqrt(((Y[A] - Y[j]) ** 2).sum())
        print('  dot %d and %s: %.4f apart on the page, pull %+.2e (above 0: together)' % (A, name, dist, 4 * (P[A, j] - Q[A, j]) * dist))
    print('  all 599 pulls on dot %d added: it moves by (%+.2e, %+.2e) times the step size' % (A, *F[A]))
    Yb, g, kept = make_map(P, tail=False, keep=KEEP)
    print('\nevery dot moves along its pull, 1,000 times:')
    for t in KEEP:
        print('  step %4d: gap %.3f, the page is %5.2f wide' % (t, gap(P, map_shares(kept[t], tail=False)), np.ptp(kept[t][:, 0])))
    saved, gs = get('bell')
    print('  (saved map: gap %.3f, largest difference to this run %.3f)' % (gs, np.abs(saved - Yb).max()))
    Yb = saved
    print('\n' + text_map(Yb, DIGITS))
    print('\nof the 10 nearest dots of a dot, %.1f %% show the same digit (keep-every-distance map: 59.3 %%, the pictures: 91.0 %%)' % (100 * same_digit(Yb, DIGITS)))
    print('of the 10 nearest dots of a dot, %.1f %% are among its 10 nearest pictures (keep-every-distance map: 26.6 %%)' % (100 * neighbours_kept(X, Yb)))
    d = np.sqrt(distances2(Yb))
    other = np.where(DIGITS[:, None] != DIGITS[None, :], d, np.inf).min(1)
    print('but the digits touch: a dot is %.2f from the nearest dot of another digit, and the page is %.1f wide' % (other.mean(), np.ptp(Yb[:, 0])))
