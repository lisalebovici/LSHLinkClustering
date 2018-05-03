import lshlink as lsh
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from functools import reduce, lru_cache
import datetime
import pickle
import pstats
import pandas as pd
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D


def data_extend(data, k):
    r, c = data.shape
    data_extend = (reduce(lambda x, y: np.vstack((x, y)),
                          map(lambda x: data, range(k))) +
                  np.random.randn(r*c*k).reshape(r*k, c).round(1))
    return(data_extend)

iris = datasets.load_iris().data
iris = data_extend(iris, 10) * 10
iris += np.abs(np.min(iris))

l = 10
k = 100
n, d = iris.shape
C = int(np.ceil(np.max(iris))) + 1
clusters = np.arange(n)


def unary(x, C):
    nearest_x = int(np.round(x))
    return((np.r_[np.ones(nearest_x),
                  np.zeros(C-nearest_x)]))

def lsh_hash(point, C):
    res = np.concatenate(list(map(lambda x: unary(x, C), point)))
    return(res)

def get_points_in_cluster(idx, clusters, data):
    point_cluster = clusters[idx]
    same_cluster_points_idx = np.where(
        clusters == point_cluster
    )[0]
    same_cluster_points = set(
        map(tuple, data[same_cluster_points_idx, :])
    )
    return same_cluster_points

def get_point_indices(data, points):
    indices = np.where((data == points[:,None]).all(-1))[1]
    return indices

def build_hash_tables(C, d, l, k, data, clusters):
    vals = np.arange(C*d)
    n = data.shape[0]
    hash_tables = defaultdict(set)
    hash_tables_reversed = defaultdict(set)

    for i in range(l):
        I = np.random.choice(vals, k, replace = False)

        for j in range(n):
            # for every point, generate hashed point
            # and sample k bits
            p = data[j]
            hashed_point = lsh_hash(p, C)[I]
            
            # check if any other points in p's cluster are
            # already in this hash table
            # and only add point to hash table if no other
            # points from its cluster are there
            bucket = hash_tables[tuple(hashed_point)]
            cluster_points = get_points_in_cluster(j, clusters, data)
            
            # create unique bucket for each hash function
            key = tuple([i]) + tuple(hashed_point)

            if not cluster_points.intersection(bucket):
                hash_tables[key].add(tuple(p))
                hash_tables_reversed[tuple(p)].add(key)

    return hash_tables, hash_tables_reversed

def build_hash_table(C, k, data):
    
    n, d = data.shape
    vals = np.arange(C*d)
    I = np.random.choice(vals, k, replace = False)
    
    HT = defaultdict(set)
    HTR = defaultdict(set)

    for j in range(n):
        p = data[j]
        hashed_point = lsh_hash(p, C)[I]
        
        bucket = HT[tuple(hashed_point)]
        cluster_points = get_points_in_cluster(j, clusters, data)

        if not cluster_points.intersection(bucket):
            HT[tuple(hashed_point)].add(tuple(p))
            HTR[tuple(p)].add(tuple(hashed_point))
    return(HT, HTR)

def LSHLinkv1(data, A, l, k, C = None, cutoff = 1):
    # set default value for C if none is provided
    if not C:
        C = int(np.ceil(np.max(data))) + 1
    
    # initializations
    n, d = data.shape
    clusters = np.arange(n)
    unique_clusters = len(np.unique(clusters))
    num = n - 1
    Z = np.zeros((n - 1, 4))
    
    # calculate r depending on n, either:
    # 1. min dist from a random sample of sqrt(n) points
    # 2. formula below
    np.random.seed(12)
    n_samp = int(np.ceil(np.sqrt(n)))
    samples = data[np.random.choice(
        n, size = n_samp, replace = False), :]
    
    if n < 500:
        r = np.min(pdist(samples, 'euclidean'))
    else:
        r = (d * C * np.sqrt(d)) / (2 * (k + d))
    
    np.random.seed(6)
    while unique_clusters > cutoff:
        # STEP 1: Generation of hash tables
        hash_tables, hash_tables_reversed = build_hash_tables(
            C, d, l, k, data, clusters)

        # STEP 2: Nearest neighbor search for p
        for i in range(n):
            # get all of those hash tables that contain point p
            p = data[i]
            p_hashes = hash_tables_reversed[tuple(p)]

            # only proceed if p is in at least one hash table
            if hash_tables_reversed[tuple(p)]:

                # find all "similar points" to p: points that
                # share at least one hash table with p, and are
                # not in the same cluster as p
                similar_points = reduce(
                    lambda x, y: x.union(y),
                    map(lambda x: hash_tables[x], p_hashes)
                    ).difference(
                    get_points_in_cluster(i, clusters, data)
                )
                similar_points = np.array(list(similar_points))

                # STEP 3: Connect pairs of clusters within certain
                # distance of p; only proceed if p has any similar points
                if similar_points.size:

                    # find similar points q s.t. dist(p, q) < r
                    # the clusters containing these points will
                    # be merged with p's cluster
                    points_to_merge = similar_points[
                        np.where(np.linalg.norm(
                            p - similar_points, axis = 1
                        ) < r)[0]
                    ]

                    # identify which clusters contain points_to_merge
                    clusters_to_merge = clusters[np.where(
                        (iris == points_to_merge[:,None]).all(-1)
                    )[1]]

                    # update cluster labels
                    clusters[np.where(
                        np.in1d(clusters,clusters_to_merge)
                    )[0]] = clusters[i]

        # STEP 4: update parameters and continue until
        # unique_clusters == cutoff
        unique_clusters = len(np.unique(clusters))

        #increase r and decrease k
        r *= A
        k = int(np.round((d * C * np.sqrt(d)) / (2 * r)))

    return(clusters)