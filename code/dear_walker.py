import numpy as np
import lshlink as lsh
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

gds10 = np.genfromtxt('../data/GDS10.csv', delimiter=",")[1:, 3:]
n = 500
gds = gds10[~np.isnan(gds10).any(axis=1)][1:n+1]

Z1 = linkage(gds, method="single")

clusters2, Z2 = lsh.LSHLink(gds, A = 2.0, l = 40, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)
clusters3, Z3 = lsh.LSHLink(gds, A = 1.6, l = 40, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)
clusters4, Z4 = lsh.LSHLink(gds, A = 1.2, l = 40, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)


C1 = cophenet(Z1)
C2 = cophenet(Z2)
C3 = cophenet(Z3)
C4 = cophenet(Z4)

print(np.corrcoef(C1, C2)[0,1])
print(np.corrcoef(C1, C3)[0,1])
print(np.corrcoef(C1, C4)[0,1])
