import numpy as np

def sarsa_lambda(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, lambda_, epsilon=0.1):
    
    # initialization
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape   
    
     
    steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
    rewards = np.zeros(num_episodes) # store total rewards for each episode
    
    for ep in range(num_episodes):
        E = np.zeros([num_states, num_actions]) # eligibility trace
        
        s = initial_state
        terminal = False   
       
        # choose action according to current policy            
        prob = np.ones(num_actions) * epsilon / num_actions
        prob[np.argmax(Q[s])] += 1 - epsilon
        a = np.random.choice(num_actions, p=prob)
            
        while not terminal:
            # take action, observe S', R
            next_s, reward, terminal = transition(s,a)

            # choose next action A'
            prob = np.ones(num_actions) * epsilon / num_actions
            prob[np.argmax(Q[next_s])] += 1 - epsilon
            next_a = np.random.choice(num_actions, p=prob)
            
            delta = reward + gamma * Q[next_s,next_a] - Q[s,a]
            E[s,a] += 1
            
            # update Q
            Q += alpha * delta * E
            
            # update E
            E *= gamma * lambda_
            
            # update s,a
            s = next_s
            a = next_a
            
            # increase number of steps in the current episode
            steps[ep] += 1
            rewards[ep] += reward
            
    return Q,  steps, rewards
        
    