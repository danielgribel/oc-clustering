import numpy as np
import time
import math
import random
from gurobipy import *

def kcenter(m, n, c, k, known_g):
    model = Model("k-center")

    z, y, x = {}, {}, {}
    
    for i in range(n):
        z[i] = model.addVar(obj=1, vtype="B", name="z[%s]"%i)
    
    for j in range(m):
        y[j] = model.addVar(obj=0, vtype="B", name="y[%s]"%j)
        for i in range(n):
            x[i,j] = model.addVar(obj=0, vtype="B", name="x[%s,%s]"%(i,j))
    
    model.update()

    for i in range(n):
        coef = [1 for j in range(m+1)]
        var = [x[i,j] for j in range(m)]
        var.append(z[i])
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign[%s]"%i)
    
    for j in range(m):
        for i in range(n):
            model.addConstr(x[i,j], "<", y[j], name = "Strong[%s,%s]"%(i,j))
    
    if known_g:
        ineq = "="
    else:
        ineq = "<"
    
    coef = [1 for j in range(m)]
    var = [y[j] for j in range(m)]
    model.addConstr(LinExpr(coef,var), ineq, rhs = k, name = "k_center")

    model.update()
    model.__data = x,y,z
    return model

def solve_kcenter(m, n, d, k, dist, known_g):
    model = kcenter(m, n, d, k, known_g)
    x,y,z = model.__data

    nodes = []
    edges = []
    
    LB = 0
    UB = len(dist)-1

    while LB <= UB:        
        start = time.time()
        theta = int((UB + LB) / 2)
        print 'lb, ub, theta: ', LB, UB, theta
        for j in range(m):
            for i in range(n):
                if d[i,j] > dist[theta]:
                    x[i,j].UB = 0
                else:
                    x[i,j].UB = 1.0
        
        model.update()
        model.Params.OutputFlag = 0 # silent mode
        model.optimize()
        
        infeasibility = sum([z[i].X for i in range(m)])

        if infeasibility > 0:
            LB = theta + 1
        else:
            UB = theta - 1
            nodes = [j for j in y if y[j].X > .5]
            edges = [(i,j) for (i,j) in x if x[i,j].X > .5]

        print 'time it = ', time.time() - start

    return nodes, edges

def euclidean_distance(x, y):
    return math.sqrt(sum(math.pow(a-b,2) for a, b in zip(x, y)))

def jaccard_distance(vecA, vecB):
    j = 1.0*(manhattan_distance(vecA, vecB))/len(vecA)
    return j

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
            # Note: If non-binary vectors, use the module of subtraction
            dist += 1

    return dist

# Obtains data from file reading line by line
# Each value in a line that is either 0 or 1 will be added to
# the dataset
def load_dataset(filename = 'inst.b'):
    f = open(filename,'r')
    dataset = [];
    
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

if __name__ == "__main__":
    #dataset = load_dataset()
    dataset, label = load_data("test-inst.b")
    
    n = len(dataset)
    m = n
    known_g = True
    k = 3
    delta = 1.e-4
    d = np.zeros((n,n))
    dist = set()
    z = 0

    if known_g == False:
        k = int(math.sqrt(n/2))
    
    d = np.zeros((n,n))
    
    start = time.time()

    for i in range(0, n):
        for j in range(0, n):
            d[i][j] = manhattan_distance(dataset[i], dataset[j])
            dist.add(d[i][j])

    z = max(dist)
    s = sorted(dist)

    centers, edges = solve_kcenter(m, n, d, k, s, known_g)
    
    print "Centers:", centers
    #print "Edges:", edges
    #print [((i,j),d[i,j]) for (i,j) in edges]
    print "z:", max([d[i,j] for (i,j) in edges])
    
    allocation = np.zeros(n)

    if known_g:
        correct = 0
        for e in edges:
            if label[e[0]] == centers.index(e[1]):
                correct = correct+1

        print 'acc =', (1.0*correct)/n
    
    print 'time:', time.time() - start