import numpy as np


class MonteCarloGLIEControl():
    def __init__(self, env, N0, gamma):
        # allocate Q table and N table for GLIE algorithm
        self.number_action = env.action_space.n
        self.state_space = list(env.observation_space.nvec)
        self.env = env

        # [dealer , player, action] -> [22, 11, 2]
        shape_needed = self.state_space + [self.number_action]
        self.Q = np.zeros(shape=shape_needed, dtype=np.float)
        self.N = np.zeros(shape=shape_needed, dtype=np.float)
        self.it = 0
        self.N0 = N0
        self.gamma = gamma

    def _get_action(self, state):
        player, dealer = state
        eps = self.N0/(self.N0 + np.sum(self.N[player, dealer, :]))
        random = np.random.uniform()
        if random < eps:
            # unifrom randomly chose one action.
            return np.random.choice(np.arange(start=0, stop=self.number_action, step=1))
        else:
            # chose best action from Q table.
            return np.argmax(self.Q[player, dealer, :])

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        # sample from environment for one episode
        episode = []
        self.it += 1
        while not done:
            action = self._get_action(state)
            self.N[state[0], state[1], action] += 1
            new_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = new_state
        # Backtrack and update Q values
        G = 0
        while episode:
            state, action, reward = episode.pop()
            G = reward + self.gamma * G
            # update a little for the Q value in the direction of new G
            # using learning rate alpha computed 1/N(S,A)
            alpha = 1/self.N[state[0], state[1], action]
            player, dealer = state
            self.Q[player, dealer, action] = self.Q[player, dealer,
                                                    action] + alpha * (G-self.Q[player, dealer, action])

    def train(self, number_episodes):
        for _ in range(number_episodes):
            self.train_one_episode()

    def get_values(self):
        return np.max(self.Q, axis=2)
