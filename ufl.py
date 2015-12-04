from gurobipy import *
import numpy as np
import math
import time
import sys


# Calculates the euclidean distance between two binary vectors
def euclidean_distance(x,y):
    return math.sqrt(sum(math.pow(a-b,2) for a, b in zip(x, y)))

# Calculates the manhattan distance between two binary vectors
def manhattan_distance(vecA, vecB):
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

    # Receives parameter inputs
    known_g = (sys.argv[1] == "True")
    num_g = int(sys.argv[2])
    fpn = str(sys.argv[3])
    
    # Loads dataset
    dataset, label = load_data(fpn)
    n = len(dataset)
    
    # Initializes timer
    start = time.time()

    m = Model()

    # Mark as true if G is known and set value to G
    # Otherwise G value is forced below
    if (known_g == False):
        num_g = int(math.sqrt(n/2))
    
    # Add variables
    x = {}
    y = {}
    d = {} # Distance matrix (not a variable)

    for j in range(n):
        x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    for i in range(n):
        for j in range(n):
            y[(i,j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t%d,%d" % (i,j))
            d[(i,j)] = euclidean_distance(dataset[i], dataset[j])
                
    m.update()

    # Add constraints
    for i in range(n):
        for j in range(n):
            m.addConstr(y[(i,j)] <= x[j])

    for i in range(n):
        q_sum = 0
        for j in range(n):
            q_sum += y[(i,j)]
        m.addConstr(q_sum == 1)

    m.addConstr(quicksum(x[i] for i in range(n)) == num_g)

    z = 0
    for i in range(n):
        for j in range(n):
            z += d[(i,j)] * y[(i,j)]    

    m.setObjective(z)

    m.optimize()

    # Stops timer
    runtime = time.time() - start
    
    # Handler for output file
    f = open(fpn + '_results_ufl_gKnownIs' + str(known_g), 'w')
    
    # Writes output
    fac = []
    num_fac = 0
    f.write( 'Facilities:\n' )
    for i in range(0,len(x)):
        if (x[i].X == 1):
            fac.append(i)
            num_fac += 1
            f.write( str(i) + '\n' )

    sumf = np.zeros(num_fac)
    f.write( 'Allocations:\n' )
    for i in range(0,len(x)):
        if (i not in fac):
            strfac = ""
            for j in range(0,num_fac):
                sumf[j] += y[i,fac[j]].X
                strfac += 'Fac'+str(j)+' = ' + str(y[i,fac[j]].X) + ' '
            f.write( strfac + ' i = ' + str(i) + '\n')

    for j in range(0,num_fac):
        f.write( 'Sum of Allocations to F'+str(j)+': ' + str(sumf[j]) + '\n' )

    f.write( 'n = ' + str(n) + '\n' )
    if (known_g == True):
        f.write( 'g = ' + str(num_g) + '\n' )
        f.write( 'g is known\n' )
    else:
        f.write( 'num fac = ' + str(num_fac) + '\n' )
        f.write( 'g not known\n' ) 
        
    f.write( 'Time (IO and Dataset Loading not included) = ' + str(runtime) + '\n' )
    
    edges = [(i,j) for (i,j) in y if y[i,j].X > 0]
    centers = [j for j in x if x[j].X > 0]
    
    if known_g:
        correct = 0
        for e in edges:
            if label[e[0]] == centers.index(e[1]):
                correct = correct + 1

        f.write( 'acc =' + str((1.0 * correct) / n) + '\n' )
    
    f.close()







