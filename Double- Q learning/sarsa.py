import numpy as np
import random


def sarsa(initial_Q, initial_state, transition,
          num_episodes, gamma, alpha, epsilon=0.1):
    """
    This function implements Sarsa. It returns learned Q values.
    To crete Figure 6.3 and 6.4, the function also returns number of steps, and
    the total rewards in each episode.

    Notes on inputs:
    -transition: function. It takes current state s and action a as parameters
                and returns next state s', immediate reward R, and a boolean
                variable indicating whether s' is a terminal state.
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy

    """

    # initialization
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape

    steps = np.zeros(num_episodes, dtype=int)  # store #steps in each episode
    rewards = np.zeros(num_episodes)  # store total rewards for each episode

    # my code
    def get_action(num_actions, Q, cur_state):
        rand_num = random.random()
        pon_action = np.argmax(Q[cur_state])
        if rand_num > (1 - epsilon):
            action = random.randint(0, num_actions - 1)
        else:
            action = pon_action
        return action

    for ep in range(num_episodes):
        """
        your code
        """
        cur_state = initial_state
        step = 0
        total_reward = 0
        terminal = False
        cur_action = get_action(num_actions, Q, cur_state)
        while not terminal:
            next_state, reward, terminal = transition(cur_state, cur_action)
            next_action = get_action(num_actions, Q, next_state)
            Q[cur_state, cur_action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[cur_state, cur_action])
            cur_state = next_state
            cur_action = next_action
            total_reward += reward
            step += 1
        steps[ep] = step
        rewards[ep] = total_reward
    return Q, steps, rewards


"""
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
"""