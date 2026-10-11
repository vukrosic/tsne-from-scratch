"""Step 1. A picture is 64 numbers. How far apart are two pictures?"""
import numpy as np
from tsne import pictures, distances2, nearest

X, digits, order = pictures()
A = 36                                  # the picture the video follows: a 3
d = np.sqrt(distances2(X))              # distance between every two pictures
near = nearest(d, 10)
SEVEN = int(np.where(digits == 7)[0][0])

if __name__ == '__main__':
    print('%d pictures, %d numbers each, from %.2f (empty square) to %.2f (full ink)' % (*X.shape, X.min(), X.max()))
    print('picture %d shows a %d. Its first 8 numbers: %s' % (A, digits[A], ' '.join('%.2f' % v for v in X[A][:8])))
    b = int(near[A][0])
    print('\nits nearest picture is picture %d, a %d' % (b, digits[b]))
    for k in (2, 3, 4):
        print('  square %d: %.4f and %.4f, difference %.4f, times itself %.4f' % (k + 1, X[A][k], X[b][k], X[A][k] - X[b][k], (X[A][k] - X[b][k]) ** 2))
    total = ((X[A] - X[b]) ** 2).sum()
    print('  all 64 squares added: %.4f. The root of that: distance %.2f' % (total, np.sqrt(total)))
    print('picture %d shows a 7: distance %.2f. The farthest picture: distance %.2f' % (SEVEN, d[A, SEVEN], d[A].max()))
    print('\nthe 10 nearest pictures of picture %d show: %s' % (A, ' '.join(str(digits[j]) for j in near[A])))
    print('at distances: %s' % ' '.join('%.2f' % d[A, j] for j in near[A]))
    same = (digits[near] == digits[:, None]).mean()
    print('\nall %d pictures: of the 10 nearest pictures, %.1f %% show the same digit' % (len(X), 100 * same))
    print('a page has room for 2 numbers per dot. A picture has %d' % X.shape[1])
