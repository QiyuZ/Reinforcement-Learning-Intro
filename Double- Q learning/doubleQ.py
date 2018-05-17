import numpy as np
import random


def doubleQ(initial_Q1, initial_Q2, initial_state, transition,
            num_episodes, gamma, alpha, epsilon=0.1):
    # This function implements double Q-learning. It returns Q1, Q2 and their sum Q

    """
    Your code
    """
    Q1 = np.copy(initial_Q1)
    Q2 = np.copy(initial_Q2)
    Q = Q1 + Q2
    num_states, num_actions = Q.shape
    prob = (epsilon / num_actions) + (1 - epsilon)

    def get_action(num_actions, Q, cur_state, prob):
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
        terminal = False
        while not terminal:
            cur_action = get_action(num_actions, Q, cur_state, prob)
            next_state, reward, terminal = transition(cur_state, cur_action)
            max_a1 = np.argmax(Q1[next_state])
            max_a2 = np.argmax(Q2[next_state])
            deter = random.random()
            if deter < 0.5:
                Q1[cur_state, cur_action] += alpha * (reward + gamma * Q2[next_state, max_a1] - Q1[cur_state, cur_action])
            else:
                Q2[cur_state, cur_action] += alpha * (reward + gamma * Q1[next_state, max_a2] - Q2[cur_state, cur_action])
            Q = Q1 + Q2
            cur_state = next_state

    return Q1, Q2, Q  # ,  steps#, rewards
