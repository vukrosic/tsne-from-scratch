"""Step 3. Give up the far distances. Every picture shares out 100 % among the others."""
import numpy as np
from tsne import distances2, shares_from, neighbour_count, find_width, data_shares, COUNT
from runs import X, DIGITS
from step1_distances import A, SEVEN

if __name__ == '__main__':
    d2 = distances2(X)
    d = np.sqrt(d2)
    order = np.argsort(d[A])[1:]
    print('picture %d (a %d), a bell of width 0.50:' % (A, DIGITS[A]))
    for width in (0.5,):
        p = shares_from(d2[A], A, width)
        for rank in (1, 2, 10, 30, 100):
            j = order[rank - 1]
            print('  nearest picture number %3d: a %d at distance %.2f gets %6.2f %%' % (rank, DIGITS[j], d[A, j], 100 * p[j]))
        print('  the 7 at distance %.2f gets %.4f %%. All %d shares add up to %.0f %%' % (d[A, SEVEN], 100 * p[SEVEN], len(X) - 1, 100 * p.sum()))
    print('\nthe neighbour count of a row of shares: 30 equal shares count as %.0f, one share of 100 %% counts as %.0f' % (
        neighbour_count(np.full(30, 1 / 30)), neighbour_count(np.array([1.0]))))
    for width in (0.3, 0.5, 0.8):
        print('  width %.1f: neighbour count %.1f' % (width, neighbour_count(shares_from(d2[A], A, width))))
    w = find_width(d2[A], A)
    print('  the program tries widths until the count is %d: width %.3f' % (COUNT, w))
    P, widths = data_shares(X, give_widths=True)
    lo, hi = int(widths.argmin()), int(widths.argmax())
    for name, i in (('narrowest', lo), ('widest', hi)):
        s = np.sort(d[i])
        print('%s bell: picture %d, width %.3f. Its nearest picture is %.2f away, its 30th nearest %.2f' % (name, i, widths[i], s[1], s[30]))
    print('\none table for all pairs: %d pairs, the shares add up to %.0f %%' % (len(X) * (len(X) - 1) // 2, 100 * P.sum()))
    even = 1.0 / (len(X) * (len(X) - 1))
    print('an even split would give every pair %.5f %%. %d pairs get more, %d pairs get less' % (
        200 * even, (P > even).sum() // 2, (P <= even).sum() // 2 - len(X) // 2))
