"""Step 2. The simple try: draw every distance right. A page has no room for that."""
import numpy as np
from tsne import distances2, same_digit, neighbours_kept, nearest
from runs import X, DIGITS, get


def best_square():
    """Four pictures that are all 1 apart. The best four dots on a page: a square."""
    side = (4 + 2 * np.sqrt(2)) / 8     # the side that makes the total error smallest
    return side, side * np.sqrt(2)


if __name__ == '__main__':
    print('3 pictures, every pair 1 apart: a triangle with sides 1, 1, 1. Every distance is right.')
    side, across = best_square()
    print('4 pictures, every pair 1 apart: the best 4 dots are a square. Sides %.2f, corner to corner %.2f.' % (side, across))
    print('  no 4 dots on a page are all 1 apart. With 64 numbers, 65 pictures can be.')
    Y, error = get('keep')
    d, m = np.sqrt(distances2(X)), np.sqrt(distances2(Y))
    print('\n600 dots moved until the distances on the page fit the distances of the pictures as well as they can:')
    print('  a distance on the page is wrong by %.0f %% on average' % error)
    n1 = nearest(distances2(X), 1)[:, 0]
    rows = np.arange(len(X))
    print('  a picture and its nearest picture: %.2f apart in the 64 numbers, %.2f apart on the page' % (d[rows, n1].mean(), m[rows, n1].mean()))
    print('  of the 10 nearest dots of a dot, %.1f %% show the same digit (in the 64 numbers: 91.0 %%)' % (100 * same_digit(Y, DIGITS)))
    print('  of the 10 nearest dots of a dot, %.1f %% are among its 10 nearest pictures' % (100 * neighbours_kept(X, Y)))
