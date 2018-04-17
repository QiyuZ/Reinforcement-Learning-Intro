import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MC_eVisit import *
from blackjack_setup import *

num_episodes = 10000
#num_episodes = 50000
v_pi = MC_eVisit(get_episode_blackjack,crazyPolicy,initial_v,1,0,num_episodes)

x, y = np.meshgrid(range(1,11),range(12,22))

fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(211,projection = '3d')

ax1.plot_surface(x,y,v_pi[:100].reshape((10,10),order='F'), rstride=1,
                cstride=1,color='w')
ax1.set_title('No usable Ace, after %d episodes' % num_episodes)
ax1.set_xlabel('dealer showing')
ax1.set_ylabel('player sum')

ax2 = fig.add_subplot(212,projection = '3d')
ax2.plot_surface(x,y,v_pi[100:200].reshape((10,10),order='F'), rstride=1,
                cstride=1,color='w')
ax2.set_title('Usable Ace, after %d episodes' % num_episodes)
ax2.set_xlabel('dealer showing')
ax2.set_ylabel('player sum')

fig.savefig('blackjack_prediction_%d_episodes.jpg' %num_episodes)

