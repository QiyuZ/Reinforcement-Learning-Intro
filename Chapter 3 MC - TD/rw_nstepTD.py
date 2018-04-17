import numpy as np
import matplotlib.pyplot as plt
from randomWalk_setup import get_episode_randomWalk19
from TD import TD_n

v_true =  np.arange(-9,10) / 10

random_policy = np.ones([21,2]) / 2 
initial_v = np.zeros(21) 
gamma = 1
num_episodes = 10


alphas = np.arange(0,1.1,0.1)
runs = 20 # we will have 20 repetitions instead of 100 as in the textbook

steps = np.power(2, np.arange(4)) # n = 1,2,4,8
errors = np.zeros((len(steps),len(alphas)))

for run in range(runs):   
    for stepIndex, step in zip(range(len(steps)), steps):
        print('run: '+str(run+1)+' n= ' +str(step))        
        for alphaIndex, alpha in zip(range(len(alphas)), alphas):
            v = np.copy(initial_v)
            for ep in range(num_episodes):
                v = TD_n(get_episode_randomWalk19, random_policy, v, 
                         step, gamma, alpha)
                error = np.sqrt(np.sum( np.power((v_true - v[1:20]), 2) / 19 ))
                errors[stepIndex, alphaIndex] += error
errors /=  runs*num_episodes
errors[errors > 0.55] = 0.55

fig=plt.figure(figsize=(8,6))     
ax = fig.add_subplot(111)           
for i in range(len(steps)):
    ax.plot(alphas,errors[i,:], label='n = ' +str(steps[i]))
ax.set_xlabel('alpha')
ax.set_ylabel('RMS error')
ax.set_ylim([0.25,0.55])
lgd=ax.legend(loc='upper left',bbox_to_anchor=(1,1))
fig.savefig('randomWalk_nStepTD.jpg',bbox_extra_artists=(lgd,), 
            bbox_inches='tight',dpi=100)