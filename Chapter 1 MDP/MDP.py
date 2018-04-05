import numpy as np
from prettytable import PrettyTable

def generate_episode_studentMDP(policy,initial_state = 'C1'):
    states_str = ['Facebook','C1','C2','C3','Sleep']
    actions_str = ['relax','study']
    initial_state = states_str.index(initial_state)
    
    P = np.zeros([5,2,5]) 
    # action 1: study (or quit facebook;  action 0: relax (facebook, sleep, or pub)
    P[0,0,0] = 1
    P[0,1,1] = 1
    P[1,0,0] = 1
    P[1,1,2] = 1
    P[2,0,4] = 1
    P[2,1,3] = 1
    P[3,0,1] = 0.2
    P[3,0,2] = 0.4
    P[3,0,3] = 0.4
    P[3,1,4] = 1
    P[4,0,4] = 1
    P[4,1,4] = 1

    r = np.zeros([5,2,5]) 
    # action 1: study;  action 0: relax (facebook, sleep, or pub)
    r[0,0,0] = -1
    r[0,1,1] = 0
    r[1,0,0] = -1
    r[1,1,2] = -2
    r[2,0,4] = 0
    r[2,1,3] = -2
    r[3,0,1] = 1
    r[3,0,2] = 1
    r[3,0,3] = 1
    r[3,1,4] = 10
    
    states = [initial_state]
    current_state = initial_state
    num_states, num_actions = policy.shape
    actions = []
    rewards = []
    
    while current_state != 4:
        action = np.random.choice(range(num_actions), p = policy[current_state,:] )
        next_state = np.random.choice(range(num_states), p = P[current_state,action,:])
        reward = r[current_state,action,next_state]
        
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        current_state = next_state
    
    states = [states_str[s] for s in states]
    actions = [actions_str[a] for a in actions]

    episode = {'states':states, 'actions':actions, 'rewards':rewards}
    return episode


random_policy = np.ones([5,2])*0.5

episode = generate_episode_studentMDP(random_policy)

# print the episode
table = PrettyTable(['state', 'action','reward'])
for i in range(len(episode['actions'])):
    table.add_row([episode['states'][i], episode['actions'][i],episode['rewards'][i]])
table.add_row([episode['states'][-1],' ',' '])
print(table)   
