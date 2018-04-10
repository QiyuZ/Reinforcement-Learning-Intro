import numpy as np
from prettytable import PrettyTable
from dynamicProgramming import *

# initialization (random policy)
policy = np.zeros([16,4])
policy[1:15,:] = 0.25

# transition probabilities p(s,a,s')
P = np.zeros([16,4,16])
P[0,:,0] = 1
P[15,:,15] = 1

# for action = 0 (left)
for i in [1,2,3,5,6,7,9,10,11,13,14]:
    P[i,0,i-1] = 1
for i in [4,8,12]:
    P[i,0,i] = 1

# for action = 1 (up)
for i in range(3):
    P[i+1,1,i+1] = 1
for i in range(11):
    P[i+4,1,i] = 1

# for action = 2 (right)
for i in [1,2,4,5,6,8,9,10,12,13,14]:
    P[i,2,i+1] = 1
for i in [3,7,11]:
    P[i,2,i] = 1

# for action = 3 (down)
for i in range(11):
    P[i+1,3,i+5] = 1
for i in range(3):
    P[i+12,3,i+12]=1
    
# immediate reward R(s,s',a)
R = np.zeros([16,4,16])
for i in range(14):
    R[i+1,:,:] = -1

# set other parameters
theta = 1e-6
gamma = 1

def print_v(v):
    table = PrettyTable()
    table.field_names = ['state', 'value']
    for i in range(14):
        table.add_row([i+1, "%.1f" % v[i+1]])
    print(table)

v_3 = policyEval(policy,P,R,gamma,theta,3)
print('k=3')
print_v(v_3)

v_10 = policyEval(policy,P,R,gamma,theta,10)
print('k=10')
print_v(v_10)

v_inf = policyEval(policy,P,R,gamma,theta,1e6)
print('k=inf')
print_v(v_inf)

#k=3
#+-------+-------+
#| state | value |
#+-------+-------+
#|   1   |  -2.4 |
#|   2   |  -2.9 |
#|   3   |  -3.0 |
#|   4   |  -2.4 |
#|   5   |  -2.9 |
#|   6   |  -3.0 |
#|   7   |  -2.9 |
#|   8   |  -2.9 |
#|   9   |  -3.0 |
#|   10  |  -2.9 |
#|   11  |  -2.4 |
#|   12  |  -3.0 |
#|   13  |  -2.9 |
#|   14  |  -2.4 |
#+-------+-------+
#k=10
#+-------+-------+
#| state | value |
#+-------+-------+
#|   1   |  -6.1 |
#|   2   |  -8.4 |
#|   3   |  -9.0 |
#|   4   |  -6.1 |
#|   5   |  -7.7 |
#|   6   |  -8.4 |
#|   7   |  -8.4 |
#|   8   |  -8.4 |
#|   9   |  -8.4 |
#|   10  |  -7.7 |
#|   11  |  -6.1 |
#|   12  |  -9.0 |
#|   13  |  -8.4 |
#|   14  |  -6.1 |
#+-------+-------+
#k=inf
#+-------+-------+
#| state | value |
#+-------+-------+
#|   1   | -14.0 |
#|   2   | -20.0 |
#|   3   | -22.0 |
#|   4   | -14.0 |
#|   5   | -18.0 |
#|   6   | -20.0 |
#|   7   | -20.0 |
#|   8   | -20.0 |
#|   9   | -20.0 |
#|   10  | -18.0 |
#|   11  | -14.0 |
#|   12  | -22.0 |
#|   13  | -20.0 |
#|   14  | -14.0 |
#+-------+-------+
