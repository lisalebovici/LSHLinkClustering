# LSHLinkClustering
Authors: Walker Harrison & Lisa Lebovici

An implementation of locality-sensitive hashing for agglomerative hierarchical clustering (Koga, Ishibashi, Watanbe - 2006) for Duke STA 663 final project (Spring 2018).

To download, please run `pip install --index-url https://test.pypi.org/simple/ lshlink`

To import the library, enter `import lshlink` in a python interpreter.

The primary function in `lshlink` is `LSHLink()`, which compute cluster assignments and (optionally) a linkage matrix for a given dataset. The function documentation is given below:

**Parameters:**

`data (ndarray)`: d-dimensional array of size n

`A (float)`: increase ratio for r (must be > 1)

`l (int)`: number of hash functions to compute

`k (int)`: number of bits to sample from each hash point

`C (int)`: constant to determine size of unary representation; must be greater than maximal coordinate value of data set

`cutoff (int)`: the minimum number of clusters to return; if cutoff = 1, computes full hierarchy

`dendrogram (bool)`: if True, returns (n-1) x 4 linkage matrix; see documentation of `scipy.cluster.hierarchy.linkage()` for explanation of format

`seed1 (int)`: [optional] specify seed for sampled data to calculate r, if reproducibility is desired

`seed2 (int)`: [optional] specify seed for sampling of hashed bits, if reproducibility is desired

**Returns:**

`clusters (array)`: size n array containing cluster assignments for each point

`Z (ndarray)`: if dendrogram = True; (n-1) x 4 linkage matrix
