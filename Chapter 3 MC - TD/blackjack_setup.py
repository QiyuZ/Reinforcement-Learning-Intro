#######################################################################
# Based on code by:                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
#######################################################################

import numpy as np

#global stateSpace
stateSpace = np.zeros([200,3],dtype=int)
stateSpace[:,0] = np.tile(range(12,22),20)
stateSpace[:,1] = np.tile(np.repeat(range(1,11),10),2)
stateSpace[100:,2] = 1
stateSpace = stateSpace.tolist()
actionSpace = ['hit','stand']

crazyPolicy = np.zeros([200,2])
crazyPolicy[:,0] = 1
crazyPolicy[np.array(stateSpace)[:,0]>=20] = [0,1] #stand if playerSum >=20

randomPolicy = np.ones([200,2]) / 2

initial_v = np.zeros(200)

def get_episode_blackjack(policy, initialState=None, initialAction=None ):   
    states = []
    actions = []
    rewards = []
    
    playerSum = 0
    playerUsableAce = False
    dealerUsableAce = False
    
    # initialize player's cards
    if initialState is None:
        # generate a random initial state
        numAce = 0;
        while playerSum < 12: #always hit
            card = getCard()
            if card == 1:
                numAce += 1
                card = 11 # count ace as eleven initially
                playerUsableAce = True
            playerSum += card
            
        if playerSum > 21: # must have at least one ace
            playerSum -= 10
            if numAce == 1: # can't use ace as 11 anymore
                playerUsableAce = False
    
        # initialize dealer's cards
        dealerCard = getCard()
        dealerCardHidden = getCard()
        
        current_state = stateSpace.index([playerSum,dealerCard,playerUsableAce])
        
    else:
        # use specified initial state
        playerSum, dealerCard, playerUsableAce = stateSpace[initialState]
        dealerCardHidden = getCard()    
        current_state = initialState
  
    states.append(current_state)
    
    
    # initialize dealer's sum
    if dealerCard == 1 and dealerCardHidden != 1:
        dealerSum = 11 + dealerCardHidden
        dealerUsableAce = True
    elif dealerCard != 1 and dealerCardHidden == 1:
        dealerSum = dealerCard + 11
        dealerUsableAce = True
    elif dealerCard == 1 and dealerCardHidden == 1:
        dealerSum = 11 + 1
        dealerUsableAce = True
    else:
        dealerSum = dealerCard + dealerCardHidden
    

    playerBusted = False  
    
    # Let's play
    
    # player's turn
    while True:
        #print(stateSpace[current_state])
        if initialAction is not None:        
            action = initialAction
            initialAction = None
            #print('action: ', actionSpace[action])
            actions.append(action)      
        else:
            # choose action according to the given policy            
            #action = np.random.choice(2, p = policy[current_state])
            action = np.random.binomial(1,policy[current_state,1])            
            actions.append(action)
            
        if action == 1: # stand
            break;
        else: #hit
            newCard = getCard()
            playerSum += newCard
            #print('new card:', newCard)
        
            if playerSum > 21:
                if playerUsableAce == True:
                    playerSum -= 10
                    playerUsableAce = False
                    rewards.append(0)
                    current_state = stateSpace.index([playerSum, dealerCard, 
                                                      playerUsableAce])
                else:
                    #busted
                    rewards.append(-1)
                    playerBusted = True
                    break
            else:
                rewards.append(0)
                current_state = stateSpace.index([playerSum, dealerCard, 
                                                  playerUsableAce])
        states.append(current_state)
        
        
    if playerBusted:
        return states, actions, rewards
    
    # dealer's turn
    while True:
        while dealerSum < 17:
            dealerSum += getCard()
            
        if dealerSum > 21:
            if dealerUsableAce == True:
                dealerSum -= 10
                dealerUsableAce = False
            else:
                rewards.append(1)
                break
        #elif dealerSum > 17:
        else:
            if playerSum > dealerSum:
                rewards.append(1)
            elif playerSum == dealerSum:
                rewards.append(0)
            else:
                rewards.append(-1)
            break
      
    return states, actions, rewards

def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card