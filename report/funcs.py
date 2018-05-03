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


def mp (build_hash_table, args):
    p = multiprocessing.Pool()
    a = p.starmap(build_hash_table, args)
    p.close()

def mp_run_times(x1, y1, x2, y2, title = '', xlabel = '', ylabel = ''):
    plt.scatter(x1, y1)
    plt.plot(x1, y1, label = 'multiprocessing build_hash_tables()')
    plt.scatter(x2, y2)
    plt.plot(x2, y2, label = 'original build_hash_tables()')
    plt.legend(loc = 'upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gcf().set_size_inches(12, 6)
    plt.show();

def iris_run_times():
    lsh_A14_y = [.46, 2.37, 12.4, 20, 37.2, 56.4, 94]
    lsh_A14_x = [150, 450, 1050, 1500, 1950, 2550, 3000]

    lsh_A20_x = [150, 450, 1050, 1500, 1950, 2550, 3000]
    lsh_A20_y = [.285, 1.31, 7.15, 13, 21.8, 36.1, 58.2]

    sci_x = [150, 450, 1050, 1500, 1950, 2550, 3000]
    sci_y = [0.000215, 0.00136, 0.00996, 0.0187, 0.0341, 0.0592, 0.0843]

    single_x = [150, 450, 1050, 1500, 1950, 2550, 3000]
    single_y = [0.530, 7.13, 61, 149, 302, 630, 996]
    
    plt.scatter(lsh_A14_x, lsh_A14_y)
    plt.plot(lsh_A14_x, lsh_A14_y, label = 'LSH, A = 1.4')
    plt.scatter(lsh_A20_x, lsh_A20_y)
    plt.plot(lsh_A20_x, lsh_A20_y, label = 'LSH, A = 2.0')
    plt.scatter(sci_x, sci_y)
    plt.plot(sci_x, sci_y, label = 'single-linkage (scipy)')
    plt.scatter(single_x, single_y)
    plt.plot(single_x, single_y, label = 'single-linkage (custom)')
    plt.legend(loc = 'upper left')
    plt.xlabel('data size')
    plt.ylabel('time (seconds)')
    plt.title('Figure 4: Run Time Comparisons for Single-Linkage vs. LSH')
    plt.gcf().set_size_inches(12, 6)
    plt.show();

def kmeans(k, data):
    n = data.shape[0]
    clusters = np.zeros(n)
    updates = True
    iterations = 0
    
    # randomly initialize k cluster centers from input data
    idx = list(np.random.choice(range(n), k))
    centers = data[idx,]
    
    while updates:
        updates = False
        iterations += 1
        print('iteration: ' + str(iterations))
        counter = 0

        # assign each point to closest cluster center
        for i in range(n):
            point = data[i,]
            closest_cluster = np.argmin(np.linalg.norm(point - centers, axis = 1))

            if clusters[i] != closest_cluster:
                clusters[i] = closest_cluster
                updates = True
                counter += 1

        print('number of updates: %s' % str(counter))

        # change each cluster center to be in the middle of its points
        for j in range(k):
            cluster_idx = np.where(clusters == j)[0]
            centers[j] = np.mean(data[cluster_idx, :], axis = 0)
            
    return(clusters, centers)