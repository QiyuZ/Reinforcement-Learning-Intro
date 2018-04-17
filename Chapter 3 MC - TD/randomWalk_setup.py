import numpy as np

def get_episode_randomWalk(policy):
    # states_str = ['Left','A','B','C','D','E','Right']
    # actions_str = ['left', 'right']

    state = 3 # index of the initial state 
         
    states = [state]
    actions = []
    rewards = []
    
    while state != 0 and state !=6: #state is not 'Left' or 'Right'
    
        action = np.random.binomial(1,policy[state,1])
        if action == 1: # go right
            next_state = state + 1
        else:
            next_state = state - 1
        
        states.append(next_state)
        actions.append(action)
        rewards.append(0)
        state = next_state

    if state == 6 :
        rewards[-1] = 1

    return states, actions, rewards
    


def get_episode_randomWalk19(policy):
    state = 10 # index of the center state
    
    states = [state]
    actions = []
    rewards = []
    
    while state != 0 and state !=20: #state is not 'Left' or 'Right'
    
        action = np.random.binomial(1,policy[state,1])
        if action == 1: # go right
            next_state = state + 1
        else:
            next_state = state - 1
        
        states.append(next_state)
        actions.append(action)
        rewards.append(0)
        state = next_state

    if state == 20 :
        rewards[-1] = 1
    else:
        rewards[-1] = -1

    return states, actions, rewards