"""Step 5. One change on the page: a weight with a long tail. The crowd comes apart."""
import numpy as np
from tsne import distances2, map_shares, gap, make_map, same_digit, neighbours_kept, BOOST, BOOST_STEPS
from runs import X, DIGITS, shares, get
from draw import text_map, svg_map
from step4_map import KEEP


def room(Y, digits):
    """Mean distance from a dot to its nearest dot, and to its nearest dot of another digit."""
    d = np.sqrt(distances2(Y))
    np.fill_diagonal(d, np.inf)
    other = np.where(digits[:, None] != digits[None, :], d, np.inf)
    return float(d.min(1).mean()), float(other.min(1).mean())


if __name__ == '__main__':
    print('weight of two dots on the page:   bell      tail = 1 / (1 + distance x distance)')
    for d in (0.0, 1.0, 2.0, 3.0, 10.0, 30.0):
        print('  %4.0f apart                    %8.6f   %8.6f' % (d, np.exp(-d * d), 1.0 / (1.0 + d * d)))
    P = shares('digits', X)
    print('\nthe first %d steps use the shares of the pictures times %.0f: near pictures pull hard and find each other early' % (BOOST_STEPS, BOOST))
    Y, g, kept = make_map(P, keep=KEEP)
    for t in KEEP:
        print('  step %4d: gap %.3f, the page is %6.2f wide' % (t, gap(P, map_shares(kept[t])), np.ptp(kept[t][:, 0])))
    saved, gs = get('seed0')
    print('  (saved map: gap %.3f, largest difference to this run %.3f)' % (gs, np.abs(saved - Y).max()))
    Y = saved
    print('\n' + text_map(Y, DIGITS))
    print('\nthe program never read which digit a picture shows. The digits are written on the dots afterwards.')
    print('                              keep distances   bell on the page   tail on the page   (the pictures)')
    maps = [get('keep')[0], get('bell')[0], Y]
    print('same digit, 10 nearest dots        %5.1f %%           %5.1f %%            %5.1f %%          91.0 %%' % tuple(100 * same_digit(M, DIGITS) for M in maps))
    print('10 nearest pictures kept           %5.1f %%           %5.1f %%            %5.1f %%' % tuple(100 * neighbours_kept(X, M) for M in maps))
    for name, M in (('bell', maps[1]), ('tail', maps[2])):
        a, b = room(M, DIGITS)
        print('%s: a dot is %.2f from its nearest dot and %.2f from the nearest dot of another digit: %.1f times as far' % (name, a, b, b / a))
    svg_map(Y, DIGITS, 'files/map.svg', title='600 handwritten digits, t-SNE written from an empty file')
    print('picture of the map: files/map.svg')
