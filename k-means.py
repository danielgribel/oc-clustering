import numpy as np
import random
import time
import math
import sys

# Obtains data from file reading line by line
# Each value in a line that is either 0 or 1 will be added to
# the dataset
def load_data(inputdata):
    mydata = np.loadtxt(inputdata, dtype = np.object, delimiter = ',')
    mydata = mydata.astype(np.int)
    n = len(mydata)
    label = np.zeros(n, dtype=int)
    
    num_columns = mydata.shape[1]

    for i in range(0, n):
        label[i] = mydata[i][num_columns-1]

    mydata = mydata[:, :-1]

    return mydata, label

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return ( set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]) )

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    num_iter = 0
    while not has_converged(mu, oldmu):
        num_iter += 1
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters, num_iter)

    
# Receives as parameters:
# n, known_g, g, filename
if __name__ == "__main__":

    # Receives parameter inputs
    known_g = (sys.argv[1] == "True")
    g = int(sys.argv[2])
    c_y = int(sys.argv[3])
    fpn = str(sys.argv[4])

    # Mark as true if G is known and set value to G
    # Otherwise G value is forced below
    if (known_g == False):
        g = int(math.sqrt(n/2))

    # Loads dataset
    dataset, label = load_data(fpn)
    dataset = np.asarray(dataset)
    
    n = len(dataset)

    # Initializes timer
    start = time.time()

    # Call for K-Means
    mu, clusters, num_iter = find_centers(dataset, g)


    # Stops timer
    runtime = time.time() - start

    # Handler for output file
    f = open(fpn + '_results_k-means_gKnownIs' + str(known_g), 'w')
    
    # Displays Outputs
    for i in range(g):
        f.write ('Cluster number ' + str(i) + ':\n')
        for j in range(len(clusters[i])):        
            f.write( str(clusters[i][j]) + '\n' )

    for i in range(len(clusters)):
        f.write( 'Sum Centers ' + str(i) + '=' + str(len(clusters[i])) + '\n' )

    f.write( 'n = ' + str(n) + '\n' )
    f.write( 'g = ' + str(g) + '\n' )

    if (known_g == True):
        f.write( 'g is known\n' )
    else:
        f.write( 'g is not known\n' )

    f.write( 'Number of Iterations = ' + str(num_iter) + '\n' )
    f.write( 'Time (IO and Dataset Loading not included) = ' + str(runtime) + '\n' )
    
    # Makes calculations to print accuracy
    if known_g:
        # Who is the group belonging to a centroid
        c = len(clusters[0][0])
        group_cluster = [0] * len(clusters)
        for i in range (0,len(clusters)):
            col = np.zeros(c)
            for j in range(0,len(clusters[i])):
                for t in range(0,len(clusters[i][j])):
                    if (clusters[i][j][t] == 1):
                        col[t] += 1
            max_val = 0
            max_ind = 0
            for j in range(0,len(col)):
                if (col[j] > max_val):
                    max_val = col[j]
                    max_ind = j
            print col
            group_cluster[i] = int(max_ind / c_y) 
            #group_cluster[i] = max(group_cluster[i], (g - 1))
                    
        # Creates auxiliary array that holds index of cluster instead of groups
        '''
        index_gc = len(group_cluster) * [0]
        for i in range(0,len(group_cluster)):
            for j in range (0,len(group_cluster)):
                if (group_cluster[i] == j)
                    index_gc[i] = j
        '''
        
        correct = 0
        data_id = 0
        for i in range(0,len(clusters)):
            for j in range(0,len(clusters[i])):
                for k in range(0,n):
                    if (clusters[i][j] == dataset[k]).all():
                        if (group_cluster[i] == label[k]):
                            correct += 1          

        f.write( 'acc =' + str((1.0 * correct) /n) + '\n')
    
    f.close()










