import numpy as np

def MC_eVisit(get_episode,policy,initial_v,gamma,alpha,num_episodes=1):
    # initialization  
    num_states = policy.shape[0]
    v = np.copy(initial_v)
    N_s = np.zeros(num_states) # counter for states
   
    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy)
        num_transitions = len(rewards)
        
        for t in range(num_transitions):
            s = states[t]
            G = np.dot(rewards[t:],
                       [gamma ** i for i in range(num_transitions - t)])
            
            if alpha == 0:
                N_s[s] += 1
                v[s] += (G - v[s]) / N_s[s]
            else:
                v[s] += alpha * (G - v[s])
    return v
