import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class Deck():
    def __init__(self, *args, **kwargs):
        self.deck = np.arange(start=1, stop=11, step=1)

    def get_card(self, black=False):
        card = np.random.choice(self.deck)
        # when game starts, we randomly draw a black card.
        if black:
            return card
        if np.random.uniform() < 2.0/3.0:
            # card is black with prob 2/3
            return card
        else:
            # card is red with prob 1/3
            return -1 * card


class Easy21Env(gym.Env):
    def __init__(self):
        # User can take only 2 action. 0 - Hit or 1 - Stick.
        self.action_space = spaces.Discrete(2)
        # observation_space or state space is represented by 2 values.
        # Player's sum and Dealer's sum
        # each of the sum can take values between 0-21
        self.observation_space = spaces.MultiDiscrete([21, 21])
        self.deck = Deck()
        self.reset()

    def reset(self):
        self.dealer = self.deck.get_card(black=True)
        self.player = self.deck.get_card(black=True)
        return self._state()

    def busted(self, v):
        return (v < 1) or (v > 21)

    def _state(self):
        return [self.player, self.dealer]

    def step(self, action):
        assert self.action_space.contains(action)
        # if Hits
        if action == 0:
            self.player += self.deck.get_card()
            if self.busted(self.player):
                reward = -1
                done = True
            else:
                reward = 0
                done = False
        else:
            # Dealer Hits if dealer < 17
            while self.dealer < 17:
                self.dealer += self.deck.get_card()
            done = True
            if self.busted(self.dealer):
                reward = 1
            else:
                # Now this is an end case, check who wins. :)
                if self.dealer > self.player:
                    reward = -1
                elif self.dealer == self.player:
                    reward = 0
                else:
                    reward = 1
        return self._state(), reward, done
