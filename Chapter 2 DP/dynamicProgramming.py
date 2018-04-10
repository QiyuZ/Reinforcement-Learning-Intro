import numpy as np

def policyEval(policy, P, R, gamma, theta, max_iter=1e8):
    """
    This function implements the policy evaluation algorithm (the two-arrays 
    version, not the in-place version, see textbook 4.1) 
    It returns state value function v_pi.
    """    
    num_S, num_a = policy.shape    
    v = np.zeros(num_S) # initialize value function
    t = 0 

    while True and t < max_iter:
        v_old = np.copy(v)
        for i in range(num_S):
            v[i] =0
            for a in range(num_a):
                v[i] += policy[i,a] *(np.dot(P[i,a,:], R[i,a,:]+gamma*v_old[:]))
        delta = np.max(np.abs(v - v_old))
        t += 1
        
        if delta < theta:
            break
    
    return v
    

def policyImprv(P,R,gamma,policy,v):
    """
    This function implements the policy improvement algorithm.
    It returns the improved policy and a boolean variable policy_stable (True
    if the new policy is the same as the old policy)
    """
    num_S, num_a = policy.shape
    policy_new = np.zeros([num_S,num_a]) # initialize the new policy
    policy_stable = True
        
    for s in range(num_S):
        # compute q(s,a) for everay action a
        q_s = np.zeros(num_a)
        for a in range(num_a):
            q_s[a] = 0
            for s_next in range(num_S):
                q_s[a] += P[s,a,s_next] * (R[s,a,s_next] + gamma*v[s_next]) 
        
        # find actions that yield that largest q-value
        actions = np.argwhere(np.isclose(q_s , np.max(q_s)))
        policy_new[s,actions] = 1/len(actions)
        if not np.array_equal(policy[s,:],policy_new[s,:]):
            policy_stable = False
    return policy_new, policy_stable


def policyIteration(P,R,gamma,theta,initial_policy,max_iter=1e6):
    """
    This function implements the policy iteration algorithm.
    It returns the final policy and the corresponding state value function v.
    """
    policy_stable = False
    policy = np.copy(initial_policy)
    num_iter = 0
    
    while (not policy_stable) and num_iter < max_iter:
        num_iter += 1
        print('Policy Iteration: ', num_iter)
        # policy evaluation
        v = policyEval(policy,P,R,gamma,theta)
        # policy improvement
        policy, policy_stable = policyImprv(P,R,gamma,policy,v)
    return policy, v


def valueIteration(P,R,gamma,theta,initial_v,max_iter=1e8):
    """
    This function implements the value iteration algorithm (the in-place version).
    It returns the best action for each state  under a deterministic policy, 
    and the corresponding state-value function.
    """
    print('Running value iteration ...')    
    #v = np.copy(initial_v)
    v = initial_v    
    num_states, num_actions = P.shape[:2]
    t = 0
    best_actions = [0] * num_states
    
    while True and t < max_iter:
        t += 1
        delta = 0
        #v_old = np.copy(v)
        for s in range(num_states):
            v_old = v[s]
            q_sa = np.zeros(num_actions)
            for a in range(num_actions):
                q_sa[a] = np.dot(P[s,a,:], R[s,a,:] + gamma * v)
            v[s] = max(q_sa)
            best_actions[s] = np.argmax(q_sa)
            delta = max(delta,abs(v[s] - v_old))
        
        if delta < theta:
            break
    print('number of iterations:', t)
    return best_actions, v
