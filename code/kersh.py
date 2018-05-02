kershaw = pd.read_csv('../data/Kershaw.csv').iloc[:, 1:5]

kershaw_data = np.array(kershaw.iloc[:, 0:3])
kershaw_labels = np.array(kershaw.iloc[:, 3])
np.unique(kershaw_labels)

a = np.array([0])
b = np.min(kershaw_data, axis = 0)[1:3]
shift = np.r_[a, b]

kershaw_data_shifted = kershaw_data - shift

kersh_clusters = lsh.LSHLink(kershaw_data_shifted, A = 1.1, l = 10, k = 100, seed1 = 12, seed2 = 6, cutoff = 3)
np.unique(kersh_clusters)

TRUTH = np.c_[kershaw_data, kersh_clusters]
g1 = np.where(kersh_clusters == np.unique(kersh_clusters)[0])[0]
g2 = np.where(kersh_clusters == np.unique(kersh_clusters)[1])[0]
g3 = np.where(kersh_clusters == np.unique(kersh_clusters)[2])[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TRUTH[g1, 0], TRUTH[g1, 1], TRUTH[g1, 2], alpha = 0.1, c = 'red')
ax.scatter(TRUTH[g2, 0], TRUTH[g2, 1], TRUTH[g2, 2], alpha = 0.1, c = 'green')
ax.scatter(TRUTH[g3, 0], TRUTH[g3, 1], TRUTH[g3, 2], alpha = 0.1, c = 'blue')
ax.view_init(azim=100)
ax.set_xlabel('start_speed')
ax.set_ylabel('pfx_x')
ax.set_zlabel('pfx_z')
plt.show()