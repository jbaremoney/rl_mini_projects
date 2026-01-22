# actions are betting amounts
# start with $20, state is dollar amount
#
# The possible next states are {current + bet, current - bet} since both are equally likely,
# we get our expectation of the next state to be current state

import random
from models.finite_markov import FiniteMarkovRewards


class GamblersRemorseSim():
    def __init__(self, initial: int, goal: int, p):
        # p is p(win)
        self.initial = initial
        self.goal = goal
        self.p = p
        self.state_hist = []
        self.current = initial

    @staticmethod
    def sample():
        pass

    def reward(self, amount=None):
        if amount == None:
            if self.current > self.state_hist[-1]:
                return 2

            elif self.current < self.state_hist[-1]:
                return -1

            else:
                if self.current == 90:
                    return 1000
                else:
                    return -2

        else:
            if amount > self.current:
                return 2

            elif self.current < self.current:
                return -1

            else:
                if self.current == 90:
                    return 1000
                else:
                    return -2

    # weighted sum of probs of next states times the rewards the states correspond to.
    # in this case the next possible states are current+bet, current - bet
    def expected_reward_next(self, bet, state=None):
        if state == None:
            return self.p*self.reward(self.current + bet) + self.p*self.reward(self.current - bet)
        else:
            return self.p * self.reward(state + bet) + self.p * self.reward(state - bet)

    # betting is the action
    def bet(self):
        # get proportion of current sum to bet
        dist = 90 - self.current
        prop = 1 - (dist/90)
        return prop*self.current


