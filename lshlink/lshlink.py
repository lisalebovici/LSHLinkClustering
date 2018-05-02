import numpy as np
import matplotlib.pyplot as plt
import pickle

from collections import defaultdict
from scipy.spatial.distance import pdist
from functools import reduce, lru_cache


def singleLink(k, data):
    '''
    Computes cluster assignments for a data set using single-linkage
    agglomerative hierarchical clustering.
    
    -----------
    Parameters:
    -----------
    k (int): number of clusters to return
    data (ndarray): d-dimensional data set of size n
    
    --------
    Returns:
    --------
    clusters (array): size n array containing cluster assignments for each point
    '''

    n = data.shape[0]
    
    # start with each point in its own cluster
    clusters = np.arange(n)
    unique_clusters = len(np.unique(clusters))
    
    while unique_clusters > k:
        min_distances = np.zeros(n)
        min_points = np.zeros(n).astype('int')

        # for each point, find min distance to point not in cluster
        for i in range(n):
            point = data[i,]
            point_cluster = clusters[i]
            distances = np.linalg.norm(point - data, axis = 1)
            diff_cluster_points = np.where(clusters != point_cluster)[0]

            min_points[i] = diff_cluster_points[np.argmin(distances[diff_cluster_points])]
            min_distances[i] = distances[min_points[i]]

        # merge clusters of the two closest points
        point1_idx = np.argmin(min_distances)
        point1 = data[point1_idx,]
        point2_idx = min_points[point1_idx]
        point2 = data[point2_idx,]

        point2_cluster = clusters[point2_idx]
        clusters[np.where(clusters == point2_cluster)[0]] = clusters[point1_idx]

        # update number of clusters
        unique_clusters = len(np.unique(clusters))
    
    return clusters


@lru_cache(maxsize=None)
def unary(x, C):
    '''
    Given a coordinate value x and integer value C, computes the
    unary representation of x 1s followed by C-x 0s.
    
    -----------
    Parameters:
    -----------
    x (int): coordinate value of a data point
    C (int): constant to determine size of unary representation
    
    --------
    Returns:
    --------
    unary_rep (array): size C array containing unary representation of
        coordinate value x
    '''

    nearest_x = int(np.round(x))
    unary_rep = np.r_[np.ones(nearest_x), np.zeros(C-nearest_x)]
    return(unary_rep)


@lru_cache(maxsize=None)
def lsh_hash(point, C):
    '''
    Given a d-dimensional data point and integer value C, computes the
    hashed point using a specified unary function.
    
    -----------
    Parameters:
    -----------
    point (array): d-dimensional data point
    C (int): constant to determine size of unary representation
    
    --------
    Returns:
    --------
    res (array): size C*d array containing hashed value for data point x
    '''

    point = np.array(point)
    res = np.concatenate(list(map(lambda x: unary(x, C), point)))
    return(res)


@lru_cache(maxsize=None)
def get_points_in_cluster(idx, clusters, data):
    '''
    Finds all points in a data set that are in the same cluster
    as a given point.
    
    -----------
    Parameters:
    -----------
    idx (int): index of point within the data set
    clusters (array): cluster assignments for each point in the data set
    data (ndarray): d-dimensional data set of size n
    
    --------
    Returns:
    --------
    same_cluster_points (ndarray): multi-dimensional array of points from the
        data set that share a cluster with the specified point
    '''

    clusters = np.array(clusters)
    data = pickle.loads(data)
    point_cluster = clusters[idx]
    same_cluster_points_idx = np.where(clusters == point_cluster)[0]
    same_cluster_points = set(map(tuple, data[same_cluster_points_idx, :]))
    return(same_cluster_points)


@lru_cache(maxsize=None)
def get_point_indices(data, points):
    '''
    Given a data set and a subset of points, finds the indices
    of those points within the data set.
    
    -----------
    Parameters:
    -----------
    data (ndarray): d-dimensional data set of size n
    points (ndarray): subset of points from the data set
    
    --------
    Returns:
    --------
    indices (array): row indices of given points in the data set
    '''

    data = pickle.loads(data)
    points = pickle.loads(points)
    indices = np.where((data == points[:,None]).all(-1))[1]
    return(indices)


def build_hash_tables(C, d, l, k, data, clusters):
    '''
    Computes l hash tables for a given data set by sampling bits
    from each point's hashed unary representation.

    Only adds a point to a given hash table if no other points from
    that point's cluster already exist in the hash table.
    
    -----------
    Parameters:
    -----------
    C (int): constant which is greater than the maximal coordinate value
        in the data
    d (int): number of dimensions in the data
    l (int): number of hash functions to compute
    k (int): number of bits to sample from each hash point
    data (ndarray): d-dimensional array of size n
    clusters (array): cluster assignments for each point in the data set

    --------
    Returns:
    --------
    hash_tables (dict): hash tables where key = sampled hash point and
        value = set of points that fall into that bucket
    hash_tables_reversed (dict): reversed hash tables where key = data point
        and value = set of buckets into which that data point falls
    '''

    vals = np.arange(C*d)
    n = data.shape[0]
    hash_tables = defaultdict(set)
    hash_tables_reversed = defaultdict(set)

    for i in range(l):
        I = np.random.choice(vals, k, replace = False)

        for j in range(n):
            # for every point, generate hashed point and sample k bits
            p = data[j]
            hashed_point = lsh_hash(tuple(p), C)[I]
            
            # check if any other points in p's cluster are already in this hash table
            # and only add point to hash table if no other points from its cluster are there
            bucket = hash_tables[tuple(hashed_point)]
            cluster_points = get_points_in_cluster(j, tuple(clusters), pickle.dumps(data))

            # create unique bucket for each hash function
            key = tuple([i]) + tuple(hashed_point)

            if not cluster_points.intersection(bucket):
                hash_tables[key].add(tuple(p))
                hash_tables_reversed[tuple(p)].add(key)
    
    return(hash_tables, hash_tables_reversed)


def LSHLink(data, A, l, k, C = None, cutoff = 1, dendrogram = False, **kwargs):
    '''
    Runs locality-sensitive hashing as linkage method for agglomerative
    hierarchical clustering.

    See unary() and lsh_hash() for implementation of hash function.
    
    -----------
    Parameters:
    -----------
    data (ndarray): d-dimensional array of size n
    A (float): increase ratio for r (must be > 1)
    l (int): number of hash functions to compute
    k (int): number of bits to sample from each hash point
    C (int): constant to determine size of unary representation;
        must be greater than maximal coordinate value of data set
    cutoff (int): the minimum number of clusters to return; if cutoff = 1,
        computes full hierarchy
    dendrogram (bool): if True, returns (n-1) x 4 linkage matrix; see documentation
        of scipy.cluster.hierarchy.linkage() for explanation of format
    seed1 (int): [optional] specify seed for sampled data to calculate r,
        if reproducibility is desired
    seed2 (int): [optional] specify seed for sampling of hashed bits,
        if reproducibility is desired

    --------
    Returns:
    --------
    clusters (array): size n array containing cluster assignments for each point
    Z (ndarray): if dendrogram = True; (n-1) x 4 linkage matrix
    '''

    # set default value for C if none is provided
    if not C:
        C = int(np.ceil(np.max(data))) + 1
    
    if dendrogram and cutoff != 1:
        raise Exception('Dendrogram requires a full hierarchy; set cutoff to 1')
    
    # initializations
    n, d = data.shape
    clusters = np.arange(n)
    unique_clusters = len(np.unique(clusters))
    num = n - 1
    Z = np.zeros((n - 1, 4))
    
    # calculate r depending on n, either:
    # 1. min dist from a random sample of sqrt(n) points
    # 2. formula below

    if 'seed1' in kwargs and isinstance(kwargs['seed1'], int):
        np.random.seed(kwargs['seed1'])

    n_samp = int(np.ceil(np.sqrt(n)))
    samples = data[np.random.choice(n, size = n_samp, replace = False), :]
    
    if n < 500:
        r = np.min(pdist(samples, 'euclidean'))
    else:
        r = (d * C * np.sqrt(d)) / (2 * (k + d))
    
    if 'seed2' in kwargs and isinstance(kwargs['seed2'], int):
        np.random.seed([kwargs['seed2']])

    while unique_clusters > cutoff:
        # STEP 1: Generation of hash tables
        hash_tables, hash_tables_reversed = build_hash_tables(C, d, l, k, data, clusters)

        # STEP 2: Nearest neighbor search for p
        for i in range(n):
            # get all of those hash tables that contain point p
            p = data[i]
            p_hashes = hash_tables_reversed[tuple(p)]

            # only proceed if p is in at least one hash table
            if hash_tables_reversed[tuple(p)]:

                # find all "similar points" to p: points that share at least one
                # hash table with p, and are not in the same cluster as p
                similar_points = reduce(
                    lambda x, y: x.union(y),
                    map(lambda x: hash_tables[x], p_hashes)
                    ).difference(get_points_in_cluster(i, tuple(clusters), pickle.dumps(data)))
                similar_points = np.array(list(similar_points))

                # STEP 3: Connect pairs of clusters within certain distance of p
                # only proceed if p has any similar points
                if similar_points.size:

                    # find similar points q s.t. dist(p, q) < r
                    # the clusters containing these points will be merged with p's cluster
                    points_to_merge = similar_points[
                        np.where(np.linalg.norm(p - similar_points, axis = 1) < r)[0]
                    ]

                    # only proceed if p has similar points within distance r
                    if points_to_merge.size:
                        # identify which clusters contain points_to_merge
                        point_indices = get_point_indices(pickle.dumps(data), pickle.dumps(points_to_merge))
                        clusters_to_merge = list(np.unique(clusters[point_indices]))
                        
                        # update cluster labels
                        # if dendrogram = False, we can use a simpler method
                        if not dendrogram:
                            clusters[np.where(np.in1d(clusters, clusters_to_merge))[0]] = clusters[i]
                        
                        else:
                            clusters_to_merge.append(clusters[i])
                            
                            for j in range(len(clusters_to_merge) - 1):
                                clusterA = clusters_to_merge[j]
                                clusterB = clusters_to_merge[j+1]
                                num += 1
                                clusters[np.where(np.in1d(clusters, [clusterA, clusterB]))[0]] = num

                                Z[num - n, :] = np.array([clusterA, clusterB, r,
                                                            len(np.where(np.in1d(clusters, num))[0])])
                                clusters_to_merge[j:j+2] = 2 * [num]

        # STEP 4: update parameters and continue until unique_clusters == cutoff
        unique_clusters = len(np.unique(clusters))

        #increase r and decrease k
        r *= A
        k = int(np.round((d * C * np.sqrt(d)) / float(2 * r)))

    if not dendrogram:
        return(clusters)
    
    else:
        return(clusters, Z)


def plot_clusters(raw, cutoff, scale=1, linkage='LSH', **kwargs):
    '''
    Plots data into clusters using either locality-sensitive hashing
    or single-linkage methods for agglomerative hierarchical clustering.
    
    -----------
    Parameters:
    -----------
    raw (ndarray): d-dimensional array of size n
    cutoff (int): for LSH, the minimum number of clusters to return;
        for single-linkage, the precise number of clusters to return
    scale (float): [optional] number to scale data, if necessary
    linkage (string): specify either 'LSH' for locality-sensitive hashing
        or 'single-linkage' for single-linkage method
    A (float): increase ratio for r (must be > 1); required for LSH only
    k (int): [number of sampled bits for hash function; required for LSH only
    l (int): number of hash functions to compute; required for LSH only
    seed1 (int): [optional] specify seed for sampled data in LSHLink() to calculate r,
        if reproducibility is desired
    seed2 (int): [optional] specify seed for sampling of hashed bits in LSHLink(),
        if reproducibility is desired

    --------
    Returns:
    --------
    None
    '''

    valid = ('LSH', 'single')
    if linkage not in valid:
        raise ValueError('Linkage must be one of %s' % (valid,))

    data = raw * scale
    data += np.abs(np.min(data))

    if linkage == 'LSH':
        if not all(k in kwargs for k in ('A', 'k', 'l')):
            raise KeyError(
                "if linkage == 'LSH', must provide 'A', 'k', and 'l'"
                )

        clusters = LSHLink(
            data,
            kwargs['A'],
            kwargs['k'],
            kwargs['l'],
            cutoff=cutoff,
            seed1=5,
            seed2=6
        )

    else:
        clusters = singleLink(
            cutoff,
            data
        )

    num_clusters = len(np.unique(clusters))
    
    for i in range(num_clusters):
        x = np.where(clusters == np.unique(clusters)[i])[0]
        plt.scatter(raw[x, 0], raw[x, 1])
    plt.axis('square')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    
    return


def plot_square(data):
    '''
    Plots data in square axes.
    
    -----------
    Parameters:
    -----------
    data (ndarray): d-dimensional array of size n

    --------
    Returns:
    --------
    None
    '''

    plt.scatter(data[:, 0], data[:, 1])
    plt.axis('square')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    
    return


def clear_caches():
    '''
    Clears caches populated during a run of LSHLink.

    -----------
    Parameters:
    -----------
    None

    --------
    Returns:
    --------
    None
    '''

    unary.cache_clear()
    lsh_hash.cache_clear()
    get_points_in_cluster.cache_clear()
    get_point_indices.cache_clear()
