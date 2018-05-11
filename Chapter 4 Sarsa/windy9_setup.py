import numpy as np

stateSpace = np.zeros([70,2], dtype=int)
stateSpace[:,0] = np.tile(range(10),7)
stateSpace[:,1] = np.tile(np.repeat(range(7),10),1)
stateSpace = stateSpace.tolist()
terminal_state = [7,3]

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

def transition(state,action):
    # get coordinates of the current state
    state_x, state_y = stateSpace[state]
    
    if action == 0: # left
        next_state = [max(state_x-1,0), min(state_y + wind[state_x], 6)]
   
    elif action == 1: # up
        next_state = [state_x, min(state_y + 1 + wind[state_x], 6)]

    elif action == 2: # right
        next_state = [min(state_x+1,9), min(state_y + wind[state_x], 6)]

    elif action == 3: # down
        next_state = [state_x, max(min(state_y - 1 + wind[state_x], 6), 0)]
   
    elif action == 4: # up/left
        next_state = [max(state_x-1,0), min(state_y + 1 + wind[state_x],6)]
    
    elif action == 5: # up/right
        next_state = [min(state_x+1,9), min(state_y + 1 + wind[state_x],6)]
        
    elif action == 6: # down/right
        next_state = [min(state_x+1,9), max(min(state_y - 1 + wind[state_x],6), 0)]
        
    elif action == 7: # down/left
        next_state = [max(state_x-1,0), max(min(state_y - 1 + wind[state_x],6), 0)]
    
    else: # stand
        next_state = [state_x, min(state_y + wind[state_x], 6)]
        
    terminal = next_state == terminal_state
    
    next_state = stateSpace.index(next_state)
    reward = -1
    
    return next_state, reward, terminal
    
'''
optimal route:
state: [0, 3]   action: down/right
state: [1, 2]   action: down/right
state: [2, 1]   action: down/right
state: [3, 0]   action: down/right
state: [4, 0]   action: up/right
state: [5, 2]   action: down/right
state: [6, 2]   action: down/right
state: [7,3]
number of steps:  7
'''