gds10 = np.genfromtxt('../data/GDS10.csv', delimiter=",")[1:, 3:]
n = 1000
gds = gds10[~np.isnan(gds10).any(axis=1)][1:n]

Z1 = linkage(gds, method="single")

start = datetime.datetime.now()
clusters2, Z2 = lsh.LSHLink(gds, A = 2.0, l = 10, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)
clusters3, Z3 = lsh.LSHLink(gds, A = 1.4, l = 10, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)
clusters4, Z4 = lsh.LSHLink(gds, A = 1.2, l = 10, k = 100, dendrogram = True, seed1 = 12, seed2 = 6)
end = datetime.datetime.now()
print(end - start)