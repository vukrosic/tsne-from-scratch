"""Film data only (not needed to run the steps): collects what the film draws.
python3 values.py -> ../src/values.json     Every map comes from files/maps.json (the same maps the step files print);
the bell run and the tail run are made here again with the dots of some steps kept, and checked against the saved maps."""
import json, os
import numpy as np
from tsne import (distances2, shares_from, neighbour_count, find_width, data_shares, map_shares, map_weights, gap, pulls,
                  make_map, spread, nearest, same_digit, neighbours_kept, COUNT)
from runs import X, DIGITS, GROUPS, GROUP_OF, NOISE, SEEDS, shares, get
from step6_lies import middles, nearest_group, clumping
from exercise import three_groups, between

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'src', 'values.json')
V = {}
px = lambda x: [int(round(float(v) * 16)) for v in x]                # a picture as 64 whole numbers, 0 (empty) to 16 (full ink)
r = lambda a, n=2: np.round(np.asarray(a, dtype=float), n).tolist()
ME, SEVEN = 36, 5                                                    # "our three" and "the seven" of the video
D = np.sqrt(distances2(X))
order = np.argsort(D[ME]); order = order[order != ME]
NEAR = int(order[0])
P = shares('digits', X)
N = len(X)
EVEN = 100.0 / (N * (N - 1) // 2)                                    # an even share, in percent: 100 % over all pairs
assert DIGITS[ME] == 3 and DIGITS[SEVEN] == 7 and NEAR == 396


def ten(Y, i=ME):
    """The 10 nearest dots of dot i on a map: who they are, which show the same digit, which are among its 10 nearest pictures."""
    nn = nearest(distances2(Y), 10)[i]
    real = set(int(v) for v in nearest(distances2(X), 10)[i])
    return dict(idx=[int(v) for v in nn], same=[int(DIGITS[v] == DIGITS[i]) for v in nn], kept=[int(int(v) in real) for v in nn])


def other(Y, i=ME):
    """Dot i: its nearest dot, and its nearest dot of another digit."""
    d = np.sqrt(distances2(Y))[i].copy(); d[i] = np.inf
    a = int(d.argmin()); d2 = np.where(DIGITS == DIGITS[i], np.inf, d); b = int(d2.argmin())
    return dict(near=a, near_d=round(float(d[a]), 2), other=b, other_d=round(float(d2[b]), 2))


def mean_other(Y):
    d = np.sqrt(distances2(Y)); np.fill_diagonal(d, np.inf)
    o = np.where(DIGITS[None, :] == DIGITS[:, None], np.inf, d)
    return round(float(d.min(1).mean()), 2), round(float(o.min(1).mean()), 2)


def scores(Y):
    return dict(same=round(100 * same_digit(Y, DIGITS), 1), kept=round(100 * neighbours_kept(X, Y), 1))


# ---------------------------------------------------------------- the pictures
V['n'] = dict(all=1797, pics=N, squares=64, pairs=N * (N - 1) // 2, count=COUNT, me=ME, near=NEAR, seven=SEVEN,
              per_digit=[int((DIGITS == k).sum()) for k in range(10)])
V['pics'] = [px(x) for x in X]
V['digits'] = [int(v) for v in DIGITS]
V['means'] = [px(X[DIGITS == k].mean(0)) for k in range(10)]         # the average picture of every digit

# ---------------------------------------------------------------- step 1: distances
V['dist'] = dict(d=r(D[ME], 3), near=round(float(D[ME, NEAR]), 2), seven=round(float(D[ME, SEVEN]), 2), far=round(float(D[ME].max()), 2),
                 sum=round(float(((X[ME] - X[NEAR]) ** 2).sum()), 4),
                 ten=dict(idx=[int(v) for v in order[:10]], d=r(D[ME, order[:10]], 2), digit=[int(DIGITS[v]) for v in order[:10]]),
                 same=round(100 * float((DIGITS[nearest(distances2(X), 10)] == DIGITS[:, None]).mean()), 1),
                 rings=[int((D[ME] < 1.5).sum()) - 1, int(((D[ME] >= 1.5) & (D[ME] < 2)).sum()), int(((D[ME] >= 2) & (D[ME] < 3)).sum()), int((D[ME] >= 3).sum())],
                 order=[int(v) for v in order])

# ---------------------------------------------------------------- step 2: keep every distance
Yk, err = get('keep')
dk = np.sqrt(distances2(Yk))
V['keep'] = dict(dots=r(Yk, 3), err=round(float(err)), me=ten(Yk), **scores(Yk),
                 pair=dict(pics=round(float(D[ME, NEAR]), 2), page=round(float(dk[ME, NEAR]), 2)))

# ---------------------------------------------------------------- step 3: shares
d2 = distances2(X)
trace, lo, hi = [], 1e-4, 100.0
for _ in range(60):
    mid = (lo + hi) / 2.0
    c = neighbour_count(shares_from(d2[ME], ME, mid))
    trace.append([round(mid, 4), round(c, 1)])
    if c > COUNT: hi = mid
    else: lo = mid
_, widths = data_shares(X, give_widths=True)
row = shares_from(d2[ME], ME, widths[ME])
back = shares_from(d2[NEAR], NEAR, widths[NEAR])
V['shares'] = dict(width=round(float(widths[ME]), 3), row=r(row * 100, 4), trace=trace[:18],
                   counts={str(w): round(neighbour_count(shares_from(d2[ME], ME, w)), 1) for w in (0.3, 0.5, 0.8)},
                   rows={str(w): r(shares_from(d2[ME], ME, w) * 100, 4) for w in (0.3, 0.8)},
                   near=round(float(row[NEAR]) * 100, 1), tenth=round(float(row[order[9]]) * 100, 1), seven=round(float(row[SEVEN]) * 100, 4),
                   back=round(float(back[ME]) * 100, 1), widths=r(widths, 3), wmin=round(float(widths.min()), 2), wmax=round(float(widths.max()), 2),
                   narrow=int(widths.argmin()), wide=int(widths.argmax()), even=EVEN,
                   times_near=round(float(P[ME, NEAR]) * 200 / EVEN), times_seven=round(EVEN / (float(P[ME, SEVEN]) * 200)),
                   above=int((2 * P > EVEN / 100).sum() // 2))

# ---------------------------------------------------------------- steps 4 and 5: the two runs, with the dots of some steps kept
KEEP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 130, 170, 250, 400, 650, 1000]


def run(tail, saved):
    Y, g, kept = make_map(P, tail=tail, keep=KEEP)
    Ys, gs = get(saved)
    assert abs(g - gs) < 1e-3 and np.abs(Y - Ys).max() < 1e-2, saved
    near_d, other_d = mean_other(Y)
    return dict(steps=KEEP, dots=[r(kept[t], 3) for t in KEEP], gaps=[round(gap(P, map_shares(kept[t], tail)), 3) for t in KEEP],
                wide=[round(float(np.ptp(kept[t][:, 0])), 2) for t in KEEP], gap=round(float(g), 2), me=ten(Y), near=other(Y),
                near_d=near_d, other_d=other_d, **scores(Y))


V['bell'] = run(False, 'bell')
V['tail'] = run(True, 'seed0')
Y0 = np.array(V['bell']['dots'][0])
Q0 = map_shares(Y0, False)
F, _ = pulls(P, Y0, False)
w = (P - Q0)[ME][:, None] * (Y0 - Y0[ME])                            # the pull of every partner on dot ME (up to the factor 4)
big = np.argsort(-np.abs(w).sum(1))[:14]
V['start'] = dict(near=[round(float(P[ME, NEAR]) * 200 / EVEN, 1), round(float(Q0[ME, NEAR]) * 200 / EVEN, 2)],     # in even shares: pictures, page
                  seven=[round(float(P[ME, SEVEN]) * 200 / EVEN, 3), round(float(Q0[ME, SEVEN]) * 200 / EVEN, 2)],
                  gap=V['bell']['gaps'][0], wide=V['bell']['wide'][0],
                  pulls=[[int(j), float(w[j, 0]), float(w[j, 1])] for j in big], total=[float(F[ME, 0]), float(F[ME, 1])])
V['curves'] = dict(bell3=float(np.exp(-9.0)), one_in=round(1 / float(np.exp(-9.0))), tail3=0.1, tail10=round(1 / 101, 4), times=round(0.1 / float(np.exp(-9.0))))

# ---------------------------------------------------------------- step 6, lie 1: sizes
a, b = GROUP_OF == 0, GROUP_OF == 1
Ys, _ = get('sizes')
_, gw = data_shares(GROUPS, give_widths=True)
Yt, _ = get('seed0')
V['sizes'] = dict(dots=r(Ys, 3), group=[int(v) for v in GROUP_OF], pics=[[px(x) for x in GROUPS[:12]], [px(x) for x in GROUPS[300:312]]],
                  base=[px(X[np.where(DIGITS == 3)[0][0]]), px(X[np.where(DIGITS == 7)[0][0]])],
                  mean=[px(GROUPS[a].mean(0)), px(GROUPS[b].mean(0))],
                  spread=[round(spread(GROUPS, a), 2), round(spread(GROUPS, b), 2)], map=[round(spread(Ys, a), 2), round(spread(Ys, b), 2)],
                  width=[round(float(gw[a].mean()), 3), round(float(gw[b].mean()), 3)],
                  real=[round(spread(X, DIGITS == 0), 2), round(spread(X, DIGITS == 1), 2)], real_map=[round(spread(Yt, DIGITS == 0), 1), round(spread(Yt, DIGITS == 1), 1)])


def squares(base, copy):
    """Every inked square of a copy that got bigger and stayed below 1: [square, before, change, after], with before + change = after as written."""
    out = []
    for k in range(64):
        u, v = round(float(base[k]), 2), round(float(copy[k]), 2)
        c = round(v - u, 2)
        if base[k] > 0.3 and 0.0 < copy[k] < 1.0 and c > 0 and abs(round(float(copy[k] - base[k]), 2) - c) < 1e-9:
            out.append([k, u, c, v])
    return out

def example_pair(small, big, shown=12):
    """One square of a tight copy and one of a loose copy whose change is exactly ten times as big; the small change as near to 0.02 as possible.
    Looks through the first `shown` copies of each group. Gives ([square, before, change, after] twice, [which tight copy, which loose copy])."""
    best = None
    for i in range(shown):
        for a in squares(small[0], small[1][i]):
            for j in range(shown):
                for b in squares(big[0], big[1][j]):
                    if round(b[2] - 10 * a[2], 2) == 0 and (best is None or abs(a[2] - 0.02) < abs(best[0][0][2] - 0.02)):
                        best = ([a, b], [i, j])
    return best

def far_from_middle(Z, group):
    """The distance of every member of a group from the middle of the group (their mean is the spread)."""
    return np.sqrt(((Z[group] - Z[group].mean(0)) ** 2).sum(1))


def typical_row(g, top=60):
    """The member of group g whose shares (nearest first) are closest to the average of its group: (index, its `top` biggest shares in percent)."""
    idx = np.where(GROUP_OF == g)[0]
    d2g = distances2(GROUPS)
    rows = np.array([np.sort(shares_from(d2g[i], i, gw[i]))[::-1][:top] for i in idx])
    k = int(np.abs(rows - rows.mean(0)).sum(1).argmin())
    return int(idx[k]), rows[k] * 100


t0, t1 = typical_row(0), typical_row(1)
ex = example_pair((X[ME], GROUPS[:300]), (X[SEVEN], GROUPS[300:]))
V['sizes'].update(example=ex[0], example_copy=ex[1],                # the second change: ten times the first, as in the video
                  far=[r(far_from_middle(GROUPS, a), 3), r(far_from_middle(GROUPS, b), 3)],
                  row_of=[t0[0], t1[0]], row=[r(t0[1], 2), r(t1[1], 2)],
                  real_far=[r(far_from_middle(X, DIGITS == 0), 3), r(far_from_middle(X, DIGITS == 1), 3)])
assert int(np.where(DIGITS == 3)[0][0]) == ME and int(np.where(DIGITS == 7)[0][0]) == SEVEN      # the two copied pictures are "our three" and "the seven"

# ---------------------------------------------------------------- lie 2: five starts
truth = nearest_group(X, DIGITS)
maps = [get('seed%d' % s) for s in SEEDS]
n0, n1 = nearest(distances2(maps[0][0]), 10), nearest(distances2(maps[1][0]), 10)
V['starts'] = dict(dots=[r(Y, 2) for Y, _ in maps], gaps=[round(float(g), 2) for _, g in maps], mid=[r(middles(Y, DIGITS), 2) for Y, _ in maps],
                   near=[[int(v) for v in nearest_group(Y, DIGITS)] for Y, _ in maps], truth=[int(v) for v in truth],
                   right=[int((nearest_group(Y, DIGITS) == truth).sum()) for Y, _ in maps],
                   stay=round(100 * float(np.mean([len(set(n0[i]) & set(n1[i])) / 10.0 for i in range(N)]))),
                   me=[[int(v) for v in n0[ME]], [int(v) for v in n1[ME]]])
md = np.sqrt(distances2(middles(X, DIGITS)))                         # distances between the ten average pictures
stay10 = [len(set(n0[i]) & set(n1[i])) for i in range(N)]            # of a dot's 10 nearest dots on the first map, how many are among its 10 nearest on the second
room = [min(float(np.sqrt(distances2(Y))[i, nn[i][-1]] / np.abs(Y).max()) for (Y, _), nn in zip(maps[:2], (n0, n1))) for i in range(N)]
inside = lambda i: DIGITS[i] == 0 and all(DIGITS[v] == 0 for v in list(n0[i]) + list(n1[i]))       # a zero whose near dots are all zeros, on both maps
one = max((i for i in range(N) if stay10[i] == 9 and inside(i)), key=lambda i: room[i])     # the dot drawn in the video: 9 of its 10 stay, and they are easy to see
V['starts'].update(truth_d=r(md[0], 2), one=dict(idx=int(one), digit=int(DIGITS[one]), a=[int(v) for v in n0[one]], b=[int(v) for v in n1[one]]),
                   stay_hist=[int(sum(1 for v in stay10 if v == k)) for k in range(11)])

# ---------------------------------------------------------------- lie 3: the neighbour count
V['counts'] = {}
for c in (2, 30, 300):
    Y, g = get('seed0' if c == COUNT else 'count%d' % c)
    V['counts'][str(c)] = dict(dots=r(Y, 2), gap=round(float(g), 2), me=ten(Y), **scores(Y))
dn = np.sqrt(distances2(NOISE)); np.fill_diagonal(dn, np.inf)
flat = np.random.RandomState(0).rand(600, 2)
V['noise'] = dict(pics=[px(x) for x in NOISE[:24]], near=round(float(dn.min(1).mean()), 1), far=round(float(np.where(np.isinf(dn), 0, dn).max(1).mean()), 1),
                  flat=r(flat, 3), m30=r(get('noise30')[0], 2), m2=r(get('noise2')[0], 2),
                  clump=dict(flat=round(clumping(flat), 2), m30=round(clumping(get('noise30')[0]), 2), m2=round(clumping(get('noise2')[0]), 2)))

# ---------------------------------------------------------------- the exercise
Z, group = three_groups()
ex = dict(group=[int(v) for v in group], data=[round(between(Z, group, 0, 1), 1), round(between(Z, group, 0, 2), 1)], maps={})
for c in (30, 450):
    Y, g, _ = make_map(data_shares(Z, c), seed=0)
    ab, ac = between(Y, group, 0, 1), between(Y, group, 0, 2)
    ex['maps'][str(c)] = dict(dots=r(Y, 2), ab=round(ab, 1), ac=round(ac, 1), times=round(ac / ab, 2))
V['exercise'] = ex

with open(OUT, 'w') as f:
    json.dump(V, f, separators=(',', ':'))
print('wrote', os.path.getsize(OUT) // 1024, 'KB')
for k in ('n', 'keep', 'shares', 'bell', 'tail', 'start', 'curves', 'sizes', 'starts', 'noise', 'exercise'):
    v = V[k]
    print(k, {q: (w if not isinstance(w, (list, dict)) or len(json.dumps(w)) < 90 else '…') for q, w in v.items()})
print('dist', {q: w for q, w in V['dist'].items() if q not in ('d', 'order')})
print('counts', {c: {q: w for q, w in v.items() if q != 'dots'} for c, v in V['counts'].items()})
