import numpy as np

def TD0(get_episode,policy, initial_v, gamma, alpha,num_episodes = 1):
    # initialization  
    v = np.copy(initial_v)

    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy)
        num_transitions = len(rewards)
               
        for t in range(num_transitions):            
            s = states[t]
            s_next = states[t+1]
            r = rewards[t]
            v[s] += alpha * (r + gamma * v[s_next] - v[s])
    return v


def TD_n(get_episode, policy, initial_v, n, gamma, alpha,num_episodes = 1):
    v = np.copy(initial_v)
    episode = 0
    
    while episode < num_episodes:
        states, _, rewards = get_episode(policy)
        
        T = len(rewards)
        t = 0
        while True:
            tau = t - n + 1
            if tau >=0:
                s_tau = states[tau]
                lower_i = tau + 1
                upper_i = min(tau + n, T)
                G = np.dot([gamma ** (i-tau-1) for i in range(lower_i,upper_i+1)],
                            rewards[lower_i-1:upper_i] )
                if tau + n < T:
                    G = G + gamma ** n * v[states[tau+n]]
                v[s_tau] += alpha * (G - v[s_tau])
            t += 1
            
            if tau == T - 1:
                break
        episode += 1     
    return v


def TD_lambda(get_episode, policy, initial_v, lambda_, gamma, alpha,
              num_episodes=1):
    v = np.copy(initial_v)

    for ep in range(num_episodes):        
        states, _, rewards = get_episode(policy)
        T = len(rewards)
        E = np.zeros(len(v)) # initilize eligibility trace
        
        for t in range(T):
            s = states[t]
            s_next = states[t+1]
            delta = rewards[t] + gamma * v[s_next] - v[s]
            E *= gamma*lambda_
            E[s] += 1
            v += alpha*delta*E
    return v
