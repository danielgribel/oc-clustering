import numpy as np
import time
import math
import random
from gurobipy import *

def euclidean_distance(x, y):
    return math.sqrt(sum(math.pow(a-b, 2) for a, b in zip(x, y)))

# Calculates the manhattan distance between two binary vectors
def manhattan_distance(vecA, vecB):
    # Assert same length
    if (len(vecA) <> len(vecB)):
        print 'ERROR: Incompatible vector sizes'
        sys.exit(0)

    # Calculates distance
    dist = 0
    for i in range(0,len(vecA)):
        if (vecA[i] <> vecB[i]):
            dist += 1

    return dist

def jaccard_distance(vecA, vecB):
    j = 1.0*(len(vecA) - manhattan_distance(vecA, vecB))/len(vecA)
    return j

# Obtains data from file reading line by line
# Each value in a line that is either 0 or 1 will be added to the dataset
def load_dataset(filename = 'inst.b'):
    f = open(filename,'r')
    dataset = []
    
    for line in f:
        valueList = []
        for i in range (0,len(line)):
            c = line[i]
            if (c == '0'):
                valueList.append(0)
            elif (c == '1'):
                valueList.append(1)
        
        dataset.append(valueList)
    
    return dataset

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

def kmedian(m, n, c, k, known_g):
    model = Model("k-median")
    
    y = {}
    x = {}

    # variables of the model
    # x(i,j): binary. if object i is assigned to center j
    # y(j): binary. if object center j is opened
    for j in range(m):
        y[j] = model.addVar(obj=0, vtype="B", name="y[%s]"%j)
        for i in range(n):
            x[i,j] = model.addVar(obj=c[i,j], vtype="B", name="x[%s,%s]"%(i,j))

    model.update()
    
    # constraint: object must be assigned to 1 center
    for i in range(n):
        coef = [1 for j in range(m)]
        var = [x[i,j] for j in range(m)]
        model.addConstr(LinExpr(coef,var), "=", 1, name="assign[%s]"%i)

    # constraint: assign object to an opened center
    for j in range(m):
        for i in range(n):
            model.addConstr(x[i,j], "<", y[j])
    
    # constraint: (at most) k centers must be opened
    if known_g:
        ineq = "="
    else:
        ineq = "<"

    coef = [1 for j in range(m)]
    var = [y[j] for j in range(m)]
    model.addConstr(LinExpr(coef,var), ineq, rhs=k, name="k_median")          

    model.update()
    model.__data = x,y
    return model

if __name__ == "__main__":
    #dataset = load_dataset()
    dataset, label = load_data("test-inst.b")
    
    n = len(dataset)
    m = n
    known_g = False
    k = 3

    if known_g == False:
        k = int(math.sqrt(n/2))
    
    d = np.zeros((n,n))
    
    start = time.time()

    for i in range(0, n):
        for j in range(0, n):
            d[i][j] = manhattan_distance(dataset[i], dataset[j])

    model = kmedian(m, n, d, k, known_g)
    
    #model.Params.Threads = 1
    model.optimize()
    x,y = model.__data
    
    edges = [(i,j) for (i,j) in x if x[i,j].X > 0]
    centers = [j for j in y if y[j].X > 0]
    print "Opt =", model.ObjVal
    
    allocation = np.zeros(n)

    print "Centers:", centers
    #print "Edges:", edges

    if known_g:
        correct = 0
        for e in edges:
            if label[e[0]] == centers.index(e[1]):
                correct = correct+1

        print 'acc =', (1.0*correct)/n
    
    print 'time:', time.time() - start