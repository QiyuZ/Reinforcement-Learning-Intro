import numpy as np
from scipy.stats import poisson

def get_prob_and_reward(morning_cars, evening_cars, returnMean, requestMean,
                        revenue, max_cars):
    threshold = min(15,max_cars) # probability of getting a number greater than the threshold is 0

    # individual probabilities of getting returns or requests
    prob_returns = np.array([poisson.pmf(ret,mu=returnMean) 
                                 for ret in range(threshold)])
    prob_requests = np.array([poisson.pmf(req,mu=requestMean) 
                                  for req in range(threshold)])
    
    prob = 0
    reward = 0

    for requested in range(threshold):
        for returned in range(threshold):
                if evening_cars == get_evening_cars(morning_cars,
                                    requested, returned,max_cars):
                    
                    prob_temp = prob_requests[requested] * prob_returns[returned]
                    prob += prob_temp
                    reward += revenue * min(morning_cars,requested) * prob_temp
    
    if prob != 0:
        reward = reward / prob
    
    return prob,reward


def get_evening_cars(morning_cars,requested,returned,max_cars):
    return min(max(morning_cars - requested,0) + returned, max_cars)
    

def get_P_and_R(requestMean1, requestMean2, returnMean1, returnMean2,
                revenue, moving_cost, stateSpace, actions,max_cars):
    P1 = np.zeros([max_cars+1,max_cars+1])
    P2 = np.zeros([max_cars+1,max_cars+1])
    r1 = np.zeros([max_cars+1,max_cars+1])
    r2 = np.zeros([max_cars+1,max_cars+1])
    
    for m in np.arange(max_cars+1):
        for e in np.arange(max_cars+1):
            P1[m,e],r1[m,e] = get_prob_and_reward(m,e,returnMean1,requestMean1,
                                                  revenue, max_cars)
            P2[m,e],r2[m,e] = get_prob_and_reward(m,e,returnMean2,requestMean2,
                                                  revenue, max_cars)
    
    num_states = stateSpace.shape[0]
    P = np.zeros([num_states,len(actions),num_states])
    r = np.zeros([num_states,len(actions),num_states])
    
    for s in range(num_states):
        num_cars1,num_cars2 = stateSpace[s]
        
        for a, num_moved in enumerate(actions):
            if num_moved > num_cars1 or -num_moved > num_cars2 or \
                (num_moved + num_cars2) > max_cars or  \
                (num_cars1 - num_moved) > max_cars:
                P[s,a,:] = 0
                r[s,a,:] = 0
            else:
                # calculate number of cars availble next morning
                morning_cars1 = int(min(num_cars1 - num_moved, max_cars))
                morning_cars2 = int(min(num_cars2 + num_moved, max_cars))
                
                for s_next in range(num_states):
                    evening_cars1, evening_cars2 = stateSpace[s_next].astype(int)
                    prob1 = P1[morning_cars1,evening_cars1]
                    prob2 = P2[morning_cars2,evening_cars2]
                    reward1 = r1[morning_cars1,evening_cars1]
                    reward2 = r2[morning_cars2,evening_cars2]
     
                    # if morning_cars > 10, additional cost of $4 should be payed
                    if morning_cars1 > 10:
                        reward1 -= 4
                    if morning_cars2 > 10:
                        reward2 -= 4
                    
                    # one car can be shuttled for free from loc1 to loc2
                    if num_moved >=1:
                        reward1 += 2
                      
                    P[s,a,s_next] = prob1 * prob2
                    r[s,a,s_next] = reward1 + reward2 - moving_cost * abs(num_moved)
    return P, r
