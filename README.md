# t-SNE from scratch

Code for the video **How to Draw 64 Dimensions on a Flat Page, and Where the Picture Lies**.
600 small pictures of handwritten digits are 64 numbers each. The program puts one dot per picture on a flat page so that pictures that look alike get dots next to each other. It never reads which digit a picture shows, and the ten digits still end up as ten islands.
Six small programs build this from an empty file and then measure which parts of the finished map are true and which are not. About 700 lines of Python 3 with numpy. The method is from van der Maaten and Hinton, "Visualizing Data using t-SNE", 2008.

```bash
pip install numpy scikit-learn
sh run_all.sh         # six steps, the tests and the exercise; expected output is in demo-output.txt
```

Every map of the video is saved in `files/maps.json` (133 KB), so the steps only read them. All of `run_all.sh` takes about 40 seconds on a laptop: the tests and the exercise each make maps of their own.
To make every map again, delete `files/maps.json` and run `python3 runs.py` (about 2 minutes). The same random start gives the same map again (`tests.py` checks this).

There is no data to download. The pictures are the digits set that comes with scikit-learn: 1,797 handwritten digits of 8 by 8 squares. The code takes 600 of them in a fixed random order. scikit-learn is used only to load these pictures; the method itself is written in `tsne.py`.

## Words

| In the video | In the paper | In the code |
| --- | --- | --- |
| picture | datapoint x_i | 64 numbers from 0 (empty square) to 1 (full ink) |
| dot | map point y_i | 2 numbers: across and up on the page |
| distance | Euclidean distance | `distances2(X)`: the squared differences of all 64 squares, added. The root of that is the distance |
| bell | Gaussian | `shares_from(d2_row, i, width)`: a weight that falls with the distance |
| share | p_j\|i, p_ij (pictures), q_ij (page) | `data_shares(X, count)` and `map_shares(Y)`. Each table adds up to 1 |
| width of the bell | sigma_i | one width per picture, found by `find_width` |
| neighbour count | perplexity | `neighbour_count(p)`: 30 equal shares count as 30, one share of 100 % counts as 1. `COUNT = 30` |
| the new curve, heavy tail | Student t distribution with one degree of freedom | `map_weights(Y, tail=True)`: 1 / (1 + distance x distance) |
| gap | Kullback-Leibler divergence, the cost C | `gap(P, Q)`: 0 when the two tables are equal |
| pull | minus the gradient of the cost for one dot | `pulls(P, Y)` |
| step | iteration | `make_map(P, steps=1000)`: every dot moves along its pull |
| start cloud | initial solution | 600 random dots, about 0.06 wide. `seed` picks the cloud |
| island | cluster | a group of dots with empty page around it |
| spread of a group | no fixed name | `spread(Z, group)`: the mean distance of the members from the middle of the group |

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `tsne.py` | Distances, shares, gap, pulls, the map | 183 lines. Everything the six steps use. |
| `step1_distances.py` | A picture is 64 numbers; a distance between two pictures | Picture 36 is a 3. Its nearest picture is a 3 at distance 0.93. A 7 is 2.92 away. Of the 10 nearest pictures of a picture, 91.0 % show the same digit. |
| `step2_keep_distances.py` | The simple try: keep every distance | 4 pictures that are all 1 apart cannot be drawn on a page. With all 600, a distance on the page is wrong by 28 % on average, and 59.3 % of the 10 nearest dots show the same digit. |
| `step3_shares.py` | Keep only the near ones: a bell and shares | The bell width is searched per picture until the neighbour count is 30: width 0.669 for picture 36, from 0.464 to 0.865 over all pictures. One table for 179,700 pairs. |
| `step4_map.py` | Two tables, a gap, a pull on every dot | With a bell on the page too: the gap falls from 2.913 to 1.186 in 1,000 steps. 85.2 % same digit, but the digits touch. |
| `step5_tail.py` | The heavy tail | With 1 / (1 + distance x distance) on the page: gap 0.464, 94.8 % same digit, and a dot is 7.9 times as far from the nearest dot of another digit as from its nearest dot. |
| `step6_lies.py` | Which parts of the map can you trust? | Sizes: a group 8.4 times as wide in the 64 numbers is 1.01 times as wide on the map. Places: five start clouds give five different layouts of the islands, and 89.9 % of the near dots stay. Neighbour count: 2, 5, 30, 100 and 300 give five different maps, and count 2 draws clumps in pure noise. |
| `runs.py` | The saved maps | Makes every map that is missing in `files/maps.json` and reads the ones that exist. |
| `draw.py` | Maps as text and as a picture | The text maps in `demo-output.txt` and `files/map.svg`. |
| `values.py` | Numbers for the pictures of the video | Writes `../src/values.json` in the film project. Not part of `run_all.sh`; in a clone it needs a folder `../src`. |
| `tests.py` | Checks of the claims | 34 tests. One of them: a map made now is equal to the saved map. |
| `exercise.py` | Three groups on a line | See below. |

## Reading the numbers

- **Small pictures, small set.** 8 by 8 squares and 600 pictures. One dot is 0.17 % of the map. The scores are a mean over all 600 dots.
- **Two scores.** "Same digit": of the 10 nearest dots of a dot, the share that shows the same digit. "Pictures kept": of the 10 nearest dots of a dot, the share that is also among its 10 nearest pictures. The first score uses the digit names, the second does not. The program itself uses neither.
- **The settings are the ones of the paper.** 1,000 steps, step size 100, momentum 0.5 for the first 250 steps and 0.8 after, and a gain per number that grows while the direction stays the same. The video shows the pull and the steps and leaves out momentum and gain. They change how fast the map forms, not what it is pulled towards.
- **Times 4 at the start.** For the first 50 steps the shares of the pictures are multiplied by 4, so near pictures pull hard and find each other while the page is still small. The paper calls this early exaggeration. The gap printed during these steps is measured with the true shares.
- **The map with a bell on the page is the older method.** `tail=False` gives the symmetric form of SNE (Hinton and Roweis, 2002). The video uses it as the step before the heavy tail.
- **The size of the page has no unit.** The tail map is 94 wide and the bell map 7 wide. Only ratios on one map are compared, never a length on one map with a length on another.
- **The made-up groups.** 300 copies of one 3 with a random change of about 0.02 per square, and 300 copies of one 7 with a change 10 times as big (about 0.2). The spread is 0.133 and 1.114 in the 64 numbers and 7.29 and 7.33 on the map.
- **The places of the islands.** "As in the pictures" counts for how many of the ten digits the nearest island on the map is the same digit as the nearest group in the 64 numbers: 5, 4, 3, 6 and 5 of 10 for the five start clouds.
- **The noise test.** 600 pictures where every square is a random number have no groups. The measure is the distance to the nearest dot divided by the distance to the tenth nearest dot: 0.27 for dots thrown on a page at random, 0.28 for the map at neighbour count 30, 0.11 at count 2. The lower the number, the tighter the clumps.
- **Three things the video states as rules.** Near neighbours on the map can be read. The size of an island, the space between two islands and one single map cannot. These are measured here on one small set; the article by Wattenberg, Viégas and Johnson shows the same on other data.

## What this is not

- **Not a fast t-SNE.** The table has one number per pair of pictures, so the work per step grows with the square of the number of pictures. The paper's plain method is used up to about 10,000 pictures. Libraries use the tree version (van der Maaten, 2014), which treats a far group of dots as one dot.
- **Not a reproduction of the paper's figures.** The paper draws 6,000 MNIST digits of 28 by 28 squares after a first reduction to 30 numbers with PCA. Here the 64 numbers go in directly.
- **Not UMAP.** UMAP (McInnes, Healy, Melville, 2018) is a later method for the same job with another way to build the two tables. None of it is here.
- **Not a test of what a map means for other data.** The checks say what this method does to these pictures and to three made-up sets.

Sources: van der Maaten, Hinton, "Visualizing Data using t-SNE", Journal of Machine Learning Research 9, 2008. Hinton, Roweis, "Stochastic Neighbor Embedding", NIPS 2002. van der Maaten, "Accelerating t-SNE using Tree-Based Algorithms", Journal of Machine Learning Research 15, 2014. Wattenberg, Viégas, Johnson, "How to Use t-SNE Effectively", Distill, 2016. McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", arXiv 1802.03426, 2018. The pictures: `sklearn.datasets.load_digits`, from the UCI set "Optical Recognition of Handwritten Digits".

## Try it

`exercise.py` makes three groups of 200 rows of 64 numbers each. Group B is 1 away from group A. Group C is 10 away from group A, on the same line.

```python
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
```

Guess first: on the map with neighbour count 30, how many times as far from A is C as B is? Then run `python3 exercise.py` (under 10 seconds). After that guess for a neighbour count of 450 and run `python3 exercise.py 450`. Any count from 2 to 590 works. Work out why the two answers differ: each group has 200 members, and a bell set to 30 neighbours never reaches another group.
