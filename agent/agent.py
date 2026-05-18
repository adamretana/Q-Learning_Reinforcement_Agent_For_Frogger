import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        return ''.join([
            # 5 spaces in front of frog
            self.get(self.frog_x - 2, self.frog_y - 1) or '_',
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x + 2, self.frog_y - 1) or '_',

            # 2 spaces to the left
            self.get(self.frog_x - 2, self.frog_y) or '_',
            self.get(self.frog_x - 1, self.frog_y) or '_',

            # two spaces to the right
            self.get(self.frog_x + 1, self.frog_y) or '_',
            self.get(self.frog_x + 2, self.frog_y) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -100
        else:
            straight_ahead = self.key[2]
            ahead_str = self.key[:4]
            left = self.key[5:7]
            right = self.key[7:]

            reward_score = 0

            if self.frog_y == 8 or self.frog_y == 4: reward_score -= 10

            if '_' in left: reward_score += -100
            if '_' in right: reward_score += -100

            if ('>' not in ahead_str and '<' not in ahead_str) :
                reward_score += 10
            if '<' in ahead_str[2:] or '>' in ahead_str[:3] or '~' == straight_ahead:
                reward_score += -10
            if '<' not in right and '>' not in left: 
                reward_score += 5
            if '<' in right:
                reward_score += -5
            if '>' in left:
                reward_score += -5
            if '~' == straight_ahead:
                if ']' == ahead_str[1]: reward_score += 5
                elif '[' == ahead_str[3]: reward_score += 5
                else: reward_score += -2
            elif (']' == straight_ahead and ']' == ahead_str[1]) or ('[' == straight_ahead and '[' == ahead_str[3]):
                reward_score += 10
            elif '~' in ahead_str: reward_score += -10

            return reward_score
            




            


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # Current state number
        self.num = 0

        # training alpha
        self.alpha = .1

        # discount gamma
        self.gamma = .9

        # previous state info
        self.prev_action = None
        self.prev_state = None

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''

        if self.train is not None: 
            self.load()
            self.train = None

        state = Q_State(state_string)

        #print(state.key)

        try:
            state_q = self.q[state.key]
        except:
            q_init = {'u': 0, 'd': 0, 'l': 0, 'r': 0, '_': 0}
            self.q[state.key] = q_init
            state_q = self.q[state.key]

        if self.prev_state is not None:
            prev_q_state = self.q[self.prev_state.key]
            prev_q_val = prev_q_state[self.prev_action]
            prev_q_val = (1-self.alpha)*prev_q_val + self.alpha*(self.prev_state.reward() + self.gamma*state_q[max(state_q, key=state_q.get)])
            self.q[self.prev_state.key][self.prev_action] = prev_q_val

        self.save()
        self.prev_state = state

        if random.random() < self.gamma:
            max_q_action = max(state_q, key=state_q.get)
            self.prev_action = max_q_action
            return max_q_action
        else:
            action = random.choice(['u', 'd', 'l', 'r', '_'])
            self.prev_action = action
            return action
