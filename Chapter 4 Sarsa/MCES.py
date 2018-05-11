import numpy as np

def MCES(get_episode,initial_Q,initial_policy,gamma,alpha,num_episodes=1e4):
    # initialization  
    Q = np.copy(initial_Q)
    policy = np.copy(initial_policy)
    num_states, num_actions = Q.shape
    N_sa = np.zeros([num_states,num_actions]) #counter of (s,a)
    
    iteration = 0
    
    while iteration < num_episodes:
        initial_state = np.random.randint(num_states)
        initial_action = np.random.randint(num_actions)
        
        states, actions, rewards = get_episode(policy,initial_state,initial_action)
       
        num_transitions = len(rewards)
        
        for t in range(num_transitions):
            # compute G_t
            G = np.dot(rewards[t:],
                       [gamma ** i for i in range(num_transitions - t)])
                       
            s = states[t]
            a = actions[t]
            if alpha == 0:            
                N_sa[s,a] += 1
                Q[s,a] += (G - Q[s,a]) / N_sa[s,a]
            else:
                Q[s,a] += (G - Q[s,a]) * alpha
        
        # update policy
        for t in range(num_transitions):
            s = states[t]
            policy[s] =0
            policy[s,np.argmax(Q[s,:])] = 1
        iteration += 1                                        
        
    return Q , policy

