def build_hash_table(C, d, k, data):
    vals = np.arange(C*d)
    HT = defaultdict(set)
    HTR = defaultdict(set)
    I = np.random.choice(vals, k, replace = False) # figure out if replace = True or False

    for j in range(n):
            # for every point, generate hashed point and sample k bits
        p = data[j]
        hashed_point = lsh_hash(p, C)[I]
            
            # check if any other points in p's cluster are already in this hash table
            # and only add point to hash table if no other points from its cluster are there
        bucket = HT[tuple(hashed_point)]
        cluster_points = get_points_in_cluster(j, clusters, data)

        if not cluster_points.intersection(bucket):
            HT[tuple(hashed_point)].add(tuple(p))
            HTR[tuple(p)].add(tuple(hashed_point))
    return(HT, HTR)
  
  args = [] 
for i in range(l):
    args.append((C, d, k, iris))

p = multiprocessing.Pool()
a = p.starmap(build_hash_table, args)
p.close()