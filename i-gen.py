# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division

import sys
import math
import random

def oc152_t_instance_generator(n, c, c_y, p, g, n_min, n_max):
    
    if c < g * c_y:
        print ' c_y too big '
        sys.exit(0)

    num_g = []
    sum = 0
    for i in range(g):
        num_g.append(random.uniform(n_min,n_max))
        sum += num_g[i]
        print 'gen ', i, num_g[i], sum
    correct = float(n)/sum
    sum = 0
    for i in range(g):
        nr = num_g[i]*correct
        num_g[i] = int(nr)
        sum += num_g[i]
        print sum
    if sum < n:
        num_g[g-1] += 1
        
    print 'nums: ', sum, num_g
    char_g = []
    for i in range(c):
        char_g.append(-1)
        
    index = 0
    for i in range(g):
        for j in range(c_y):
            char_g[index] = i
            index += 1
            
    print ' Chars: ', char_g
    
    f = open(file_name,"w")
    
    vect = range(c)
    for i in range(g):
        print 'numg: ',num_g[i]
        for j in range(num_g[i]):
            for k in range(c):
                if char_g[k] == i:
                    vect[k] = 0
                    if random.uniform(0,1) < p:
                        vect[k] = 1
                else:
                    if char_g[k] != -1:
                        vect[k] = 0
                        if random.uniform(0,1) < 1-p:
                            vect[k] = 1
                    else:
                        vect[k] = 0
                        if random.uniform(0,1) < 0.5:
                            vect[k] = 1
            
            line = str(vect[0])
            for l in range(1,c):
                line = line + "," + str(vect[l])
            line = line + "," + str(i)

            f.write(line + "\n")

n = 99
c = 25
c_y = 5
g = 3
p = 0.85
n_min = 99
n_max = 99
file_name = 'test-inst.csv'

print  ' Params: ', n, c, c_y, p, g, n_min, n_max, file_name

oc152_t_instance_generator(n, c, c_y, p, g, n_min, n_max)