import numpy as np

def sarsa(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, epsilon=0.1):
    # initialization
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape    
    policy = np.zeros([num_states,num_actions])   
    steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
    rewards = np.zeros(num_episodes) # store total rewards for each episode
    
    for ep in range(num_episodes):
        s = initial_state
        terminal = False   
       
        # set policy based on Q-values
        policy[s] = epsilon / num_actions
        policy[s,np.argmax(Q[s])] += 1 - epsilon
        
        # choose action according to current policy            
        a = np.random.choice(num_actions, p=policy[s])
            
        while not terminal:
            # take action, observe S', R
            next_s, reward, terminal = transition(s,a)

            # choose next action A'
            policy[next_s] = epsilon / num_actions
            policy[next_s, np.argmax(Q[next_s])] += 1 - epsilon
            next_a = np.random.choice(num_actions, p=policy[next_s])
            
            # update Q(s,a)
            Q[s,a] += alpha * (reward + gamma * Q[next_s,next_a] - Q[s,a])
            
            # update s,a
            s = next_s
            a = next_a
            
            # increase number of steps in the current episode
            steps[ep] += 1
            rewards[ep] += reward
            
    return Q,  steps, rewards
        
'''
optimal route:
state: [0, 3]   action: right
state: [1, 3]   action: right
state: [2, 3]   action: right
state: [3, 3]   action: right
state: [4, 4]   action: right
state: [5, 5]   action: right
state: [6, 6]   action: right
state: [7, 6]   action: right
state: [8, 6]   action: right
state: [9, 6]   action: down
state: [9, 5]   action: down
state: [9, 4]   action: down
state: [9, 3]   action: down
state: [9, 2]   action: left
state: [8, 2]   action: left
state: [7,3]
number of steps:  15
'''
