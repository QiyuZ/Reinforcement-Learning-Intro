import numpy as np
import matplotlib.pyplot as plt
from carRentalSetup import *
from dynamicProgramming import *

requestMean1=3
requestMean2=4
returnMean1=3
returnMean2=2
max_cars=20
max_move=5
moving_cost = 2
revenue = 10

actionSpace = [i - max_move for i in range(2*max_move+1)]
num_actions = len(actionSpace)
num_states = (max_cars+1) ** 2
stateSpace = np.zeros([num_states,2])
stateSpace[:,0] = np.tile(np.arange(max_cars+1), max_cars+1)
stateSpace[:,1] = np.repeat(np.arange(max_cars+1), max_cars+1)

initial_policy = np.zeros([num_states,num_actions])
initial_policy[:,5]=1

gamma = 0.9
theta = 1e-6

print('calculating P(s,a,s'') and R(s,a,s'') ...')
P, R = get_P_and_R(requestMean1, requestMean2,
                   returnMean1, returnMean2, revenue, moving_cost,
                   stateSpace, actionSpace, max_cars)
  

actions, v = valueIteration(P,R,gamma,theta,np.zeros(num_states))

# rearrange policy and value function for easy reading and plotting 
optimal_actions = np.zeros([max_cars+1, max_cars+1],dtype=int)
optimal_value = np.zeros([max_cars+1, max_cars+1])
for s in range(num_states):
    car1, car2 = stateSpace[s].astype(int)
    optimal_actions[car1,car2] = actionSpace[actions[s]]
    optimal_value[car1,car2] = v[s]

# plot the optimal policy                                
plt.figure(figsize=(7,6))                    
plt.imshow(optimal_actions,interpolation='nearest',origin='lower',
               alpha=0.8)
plt.colorbar()
plt.xlabel('# cars at 2nd location')
plt.ylabel('# cars at 1st location')
plt.title('optimal policy from value iteration')
plt.savefig('car_rental_optimal_policy.jpg',dpi=180)


