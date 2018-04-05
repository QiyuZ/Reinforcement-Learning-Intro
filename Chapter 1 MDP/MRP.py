import numpy as np
from prettytable import PrettyTable

def generate_episode_studentMRP(initial_state = 'C1'):
    states_str = ['Facebook','C1','C2','C3','Pub','Pass','Sleep']
    initial_state = states_str.index(initial_state)
    
    # transition probablility matrix
    P = np.zeros([7,7])
    P[0][[0,1]] = [0.9,0.1]
    P[1][[0,2]] = [0.5,0.5]
    P[2][[3,6]] = [0.8,0.2]
    P[3][[4,5]] = [0.4,0.6]
    P[4][[1,2,3]] = [0.2,0.4,0.4]
    P[5][6] = 1
    P[6][6] = 1
    
    # reward for each state
    R = [-1, -2, -2, -2, 1,10, 0]
    
    states = [initial_state]
    rewards = [R[initial_state]]
    current_state = initial_state
    
    while current_state != 6:
        next_state = np.random.choice(range(7), p = P[current_state,:])
        rewards.append(R[next_state])
        states.append(next_state)
        current_state = next_state
    states = [states_str[s] for s in states]
    episode = {'states':states, 'rewards':rewards}
    return episode

# generate one episode     
episode = generate_episode_studentMRP()

# print the episode
table = PrettyTable(['state','reward'])
for i in range(len(episode['rewards'])):
    table.add_row([episode['states'][i], episode['rewards'][i]])
print(table)    
