"""Checks of the claims of the video. python3 tests.py"""
import numpy as np
from tsne import (N, COUNT, pictures, distances2, shares_from, neighbour_count, find_width, data_shares,
                  map_shares, gap, pulls, make_map, keep_distances, spread, nearest, same_digit, neighbours_kept)
from runs import X, DIGITS, GROUPS, GROUP_OF, NOISE, SEEDS, COUNTS, shares, get
from step1_distances import A, SEVEN, near, d
from step2_keep_distances import best_square
from step4_map import B
from step6_lies import nearest_group, clumping
from exercise import three_groups, between

done = []


def check(text, ok):
    done.append(bool(ok))
    print(('ok    ' if ok else 'WRONG ') + text)


def close(a, b, by=0.06):
    return abs(a - b) < by


pct = lambda v: 100.0 * v

# the pictures and their distances
check('600 pictures, 64 numbers each, every number from 0 to 1', X.shape == (600, 64) and X.min() == 0 and X.max() == 1)
n = [int((DIGITS == k).sum()) for k in range(10)]
check('every digit has 49 to 69 pictures: %s' % n, min(n) == 49 and max(n) == 69)
check('two dots at (0, 0) and (3, 4) are 5 apart: on 2 numbers the distance is the ruler',
      close(np.sqrt(distances2(np.array([[0.0, 0.0], [3.0, 4.0]]))[0, 1]), 5, 1e-9))
slow = sum((X[A][k] - X[B][k]) ** 2 for k in range(64)) ** 0.5
check('picture %d is a 3, its nearest picture %d is a 3, %.2f away. Square by square gives the same as the fast way' % (A, B, slow),
      DIGITS[A] == 3 and DIGITS[B] == 3 and close(slow, 0.93, 0.005) and close(slow, d[A, B], 1e-9))
check('its 10 nearest pictures are nine 3s and one 8; the 7 is %.2f away' % d[A, SEVEN],
      sorted(DIGITS[near[A]]) == [3] * 9 + [8] and close(d[A, SEVEN], 2.92, 0.005))
same64 = pct((DIGITS[near] == DIGITS[:, None]).mean())
check('in the 64 numbers, %.1f %% of the 10 nearest pictures show the same digit' % same64, close(same64, 91.0))

# keep every distance
side, across = best_square()
err = lambda s: 4 * (s - 1) ** 2 + 2 * (s * 2 ** 0.5 - 1) ** 2
check('4 dots for 4 pictures that are all 1 apart: a square with sides %.2f, corner to corner %.2f, has the smallest error' % (side, across),
      close(side, 0.85, 0.005) and close(across, 1.21, 0.005) and err(side) < err(side + 0.01) and err(side) < err(side - 0.01))
t = (2 ** 0.5 + 130 ** 0.5) / 128
S = np.vstack([np.eye(64) / 2 ** 0.5, np.full(64, t)])
dd = np.sqrt(distances2(S))[np.triu_indices(65, 1)]
check('65 rows of 64 numbers can be all 1 apart (%d pairs, from %.6f to %.6f)' % (len(dd), dd.min(), dd.max()), close(dd.min(), 1, 1e-6) and close(dd.max(), 1, 1e-6))
Yk, ek = get('keep')
check('keep every distance: a distance is wrong by %.0f %% on average; same digit %.1f %%, pictures kept %.1f %%' % (ek, pct(same_digit(Yk, DIGITS)), pct(neighbours_kept(X, Yk))),
      close(ek, 27.9, 0.1) and close(pct(same_digit(Yk, DIGITS)), 59.3) and close(pct(neighbours_kept(X, Yk)), 26.6))
Y2, e2 = keep_distances(X, steps=100)
check('100 steps of it made now: wrong by %.0f %% (the saved map had 600 steps)' % e2, 27 < e2 < 31)

# shares
d2 = distances2(X)
row = shares_from(d2[A], A, 0.5)
check('a row of shares adds up to 100 %% and a picture gives itself 0: %.4f' % row.sum(), close(row.sum(), 1, 1e-9) and row[A] == 0)
check('a nearer picture never gets a smaller share', bool((np.diff(row[np.argsort(d2[A])][1:]) <= 1e-15).all()))
check('with a bell of width 0.5 the nearest picture gets %.1f %% and the 7 gets less than 0.0001 %%' % pct(row[B]), close(pct(row[B]), 35.5) and pct(row[SEVEN]) < 1e-4)
check('neighbour count: 30 equal shares count as 30, 7 equal shares as 7, one share as 1',
      close(neighbour_count(np.full(30, 1 / 30)), 30, 1e-9) and close(neighbour_count(np.full(7, 1 / 7)), 7, 1e-9) and close(neighbour_count(np.array([1.0])), 1, 1e-9))
w = find_width(d2[A], A)
check('the width for picture %d is %.3f, and its shares then count as %.2f neighbours' % (A, w, neighbour_count(shares_from(d2[A], A, w))),
      close(w, 0.669, 0.001) and close(neighbour_count(shares_from(d2[A], A, w)), 30, 0.01))
P, widths = data_shares(X, give_widths=True)
check('widths from %.3f to %.3f; every picture has %d neighbours' % (widths.min(), widths.max(), COUNT),
      close(widths.min(), 0.464, 0.001) and close(widths.max(), 0.865, 0.001)
      and all(close(neighbour_count(shares_from(d2[i], i, widths[i])), 30, 0.01) for i in range(0, N, 25)))
check('the table of all pairs adds up to 100 %, is the same from both sides, and has 0 for a picture with itself',
      close(P.sum(), 1, 1e-9) and np.allclose(P, P.T) and not P.diagonal().any())

# the page
rs = np.random.RandomState(5)
Y = rs.randn(N, 2)
for tail in (False, True):
    Q = map_shares(Y, tail)
    F, _ = pulls(P, Y, tail)
    worst = 0.0
    for i, k in ((A, 0), (A, 1), (SEVEN, 0), (200, 1)):
        up, down = Y.copy(), Y.copy()
        up[i, k] += 1e-5
        down[i, k] -= 1e-5
        slope = (gap(P, map_shares(up, tail)) - gap(P, map_shares(down, tail))) / 2e-5
        worst = max(worst, abs(F[i, k] + slope) / abs(slope))
    check('%s: the shares of the page add up to 100 %%, and moving a dot along its pull makes the gap smaller (pull = minus the slope, off by %.0e)' % ('tail' if tail else 'bell', worst),
          close(Q.sum(), 1, 1e-9) and worst < 1e-4)
check('equal tables have gap 0, different tables a gap above 0', close(gap(P, P), 0, 1e-12) and gap(P, map_shares(Y)) > 0)
start = np.random.RandomState(0).randn(N, 2) * 1e-2
gb, gt = get('bell')[1], get('seed0')[1]
check('gap: %.3f at the start, %.3f with the bell on the page, %.3f with the tail' % (gap(P, map_shares(start, False)), gb, gt),
      close(gap(P, map_shares(start, False)), 2.913, 0.001) and close(gb, 1.186, 0.001) and close(gt, 0.464, 0.001))
Yb, Yt = get('bell')[0], get('seed0')[0]
check('bell: same digit %.1f %%, pictures kept %.1f %%' % (pct(same_digit(Yb, DIGITS)), pct(neighbours_kept(X, Yb))),
      close(pct(same_digit(Yb, DIGITS)), 85.2) and close(pct(neighbours_kept(X, Yb)), 49.4))
check('tail: same digit %.1f %%, pictures kept %.1f %%. The page is %.0f wide, the bell page %.0f' % (pct(same_digit(Yt, DIGITS)), pct(neighbours_kept(X, Yt)), np.ptp(Yt[:, 0]), np.ptp(Yb[:, 0])),
      close(pct(same_digit(Yt, DIGITS)), 94.8) and close(pct(neighbours_kept(X, Yt)), 67.3) and close(np.ptp(Yt[:, 0]), 94.1, 0.1) and close(np.ptp(Yb[:, 0]), 7.0, 0.1))
Yn, gn, _ = make_map(P)
check('a map made now: gap %.3f, same digit %.1f %% (other computers round differently, so the dots may sit elsewhere)' % (gn, pct(same_digit(Yn, DIGITS))),
      close(gn, 0.464, 0.03) and pct(same_digit(Yn, DIGITS)) > 92)
Ya, ga, _ = make_map(P, seed=0)
check('the same start gives the same map on this computer', np.abs(Ya - Yn).max() < 1e-9)

# the three lies
a, b = GROUP_OF == 0, GROUP_OF == 1
Ys = get('sizes')[0]
r64, rmap = spread(GROUPS, b) / spread(GROUPS, a), spread(Ys, b) / spread(Ys, a)
check('made-up groups: one is %.1f times as wide as the other in the 64 numbers and %.2f times on the map' % (r64, rmap), close(r64, 8.4) and close(rmap, 1.01, 0.02))
maps = [get('seed%d' % s) for s in SEEDS]
gaps = [g for _, g in maps]
check('five starts: gaps %s' % gaps, max(gaps) - min(gaps) < 0.01 and close(min(gaps), 0.460, 0.001))
sd = [round(pct(same_digit(Y, DIGITS)), 1) for Y, _ in maps]
check('five starts: same digit %s percent' % sd, min(sd) >= 94.5 and max(sd) <= 95.4)
ng = [list(nearest_group(Y, DIGITS)) for Y, _ in maps]
check('five starts: the island nearest to the zeros shows the digit %s' % [int(v[0]) for v in ng], [v[0] for v in ng] == [9, 5, 3, 9, 9])
truth = nearest_group(X, DIGITS)
agree = [int((np.array(v) == truth).sum()) for v in ng]
check('island neighbours that are also the nearest groups in the 64 numbers: %s of 10' % agree, agree == [5, 4, 3, 6, 5])
n0, n1 = nearest(distances2(maps[0][0]), 10), nearest(distances2(maps[1][0]), 10)
stay = pct(np.mean([len(set(n0[i]) & set(n1[i])) / 10.0 for i in range(N)]))
check('of the 10 nearest dots of a dot on map 0, %.1f %% are among its 10 nearest on map 1' % stay, close(stay, 89.9))
table = [(c, round(pct(same_digit(get('seed0' if c == COUNT else 'count%d' % c)[0], DIGITS)), 1)) for c in COUNTS]
check('neighbour count and same digit: %s' % table, [v for _, v in table] == [89.8, 95.2, 94.8, 92.8, 87.5])
flat = np.random.RandomState(0).rand(600, 2)
c30, c2 = clumping(get('noise30')[0]), clumping(get('noise2')[0])
check('nearest dot / 10th nearest dot: random dots %.2f, map of noise at count 30 %.2f, at count 2 %.2f' % (clumping(flat), c30, c2),
      close(clumping(flat), 0.27, 0.006) and close(c30, 0.28, 0.006) and close(c2, 0.11, 0.006))
Z, group = three_groups()
check('the exercise: group C is %.1f times as far from A as B is' % (between(Z, group, 0, 2) / between(Z, group, 0, 1)), close(between(Z, group, 0, 2) / between(Z, group, 0, 1), 10, 0.05))

print('\nALL TESTS PASS (%d of %d)' % (len(done), len(done)) if all(done) else '\n%d OF %d TESTS WRONG' % (done.count(False), len(done)))
