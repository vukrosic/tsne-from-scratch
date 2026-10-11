"""Draw a map without any plotting library: as text in the terminal, or as an SVG file."""
import numpy as np

COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def text_map(Y, marks=None, cols=70, rows=24):
    """Every dot becomes one character: its digit, or a star when no marks are given."""
    grid = [[' '] * cols for _ in range(rows)]
    lo, size = Y.min(0), Y.max(0) - Y.min(0) + 1e-9
    for k, (x, y) in enumerate(Y):
        c = int((x - lo[0]) / size[0] * (cols - 1))
        r = int((y - lo[1]) / size[1] * (rows - 1))
        grid[rows - 1 - r][c] = '*' if marks is None else str(marks[k])
    return '\n'.join(''.join(row).rstrip() for row in grid)


def svg_map(Y, marks, path, side=640, title=''):
    """One small digit for every dot, in the colour of that digit."""
    lo, size = Y.min(0), (Y.max(0) - Y.min(0)).max() + 1e-9
    mid = (Y.max(0) + Y.min(0)) / 2.0
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="monospace" font-size="13">' % (side, side),
           '<rect width="%d" height="%d" fill="#fbf8f1"/>' % (side, side)]
    if title:
        out.append('<text x="14" y="24" fill="#333">%s</text>' % title)
    for (x, y), m in zip(Y, marks):
        px = side / 2.0 + (x - mid[0]) / size * (side - 60)
        py = side / 2.0 - (y - mid[1]) / size * (side - 60)
        out.append('<text x="%.1f" y="%.1f" fill="%s" text-anchor="middle">%s</text>' % (px, py + 4, COLOURS[int(m) % 10], m))
    out.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(out))
