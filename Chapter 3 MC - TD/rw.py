import numpy as np
import matplotlib.pyplot as plt
from randomWalk_setup import get_episode_randomWalk
from MC_eVisit import MC_eVisit
from TD import TD0

v_True = np.array([1/6,2/6,3/6,4/6,5/6])

random_policy = np.ones([7,2]) / 2
initial_v = np.zeros(7) # value for terminal states are zero
initial_v[1:6] = 0.5

v = np.copy(initial_v)

# plot TD estimates vs true values
episodes = [0,1,10,100]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
for i in range(101):
    if i in episodes:
        ax.plot(v[1:6],label='v after ' +str(i)+' episodes')
    v = TD0(get_episode_randomWalk,random_policy, v, 1, 0.1) 
ax.plot(v_True,'k--',label = 'true values')
ax.xaxis.set_ticks(np.arange(5))
ax.xaxis.set_ticklabels(['A','B','C','D','E'])
plt.legend(loc=2)
plt.ylim([0,1])        
plt.show()
fig.savefig('figures/randomWalk_TD.jpg')


# plot RMS error for TD and MC methods, with different alpha
fig=plt.figure(figsize=(6,6))       
for alpha in [0.15,0.1,0.05]:
    print('running TD with alpha = ',alpha)
    total_errors = np.zeros(101)
    for run in range(100):
        v = np.copy(initial_v)
        errors = np.zeros(101)
        for episode in range(101):
            error = np.sqrt(np.sum( (v_True - v[1:6]) ** 2) /5 )
            errors[episode] = error
            v = TD0(get_episode_randomWalk,random_policy, v, 1, alpha)
        total_errors  += errors   
    total_errors = total_errors / 100
    plt.plot(total_errors,label='TD,alpha='+str(alpha))

for alpha in [0.01,0.02,0.03,0.04,]:
    print('running MC with alpha = ',alpha)
    total_errors = np.zeros(101)
    for run in range(100):
        v = np.copy(initial_v)
        errors = np.zeros(101)
        for episode in range(101):
            error = np.sqrt(np.sum( (v_True - v[1:6]) ** 2) /5 )
            errors[episode] = error
            v = MC_eVisit(get_episode_randomWalk, random_policy, v,1,alpha)
        total_errors  += errors   
    total_errors = total_errors / 100
    plt.plot(total_errors,'--',label='MC,alpha='+str(alpha))

plt.ylabel('Empirical RMS error')
plt.xlabel('Walks / Episodes')    
plt.legend(loc=1)
fig.savefig('figures/randomWalk_TDvsMC.jpg')