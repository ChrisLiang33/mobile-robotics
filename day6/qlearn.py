import numpy as np
class QTable:
    def __init__(self, states: np.ndarray, actions: np.ndarray, gamma=1, alpha = 0.1):
        state_size = np.size(states)
        action_size = np.size(actions)
        self.table = np.zeros((state_size,action_size))
        self.rewards = np.zeros((state_size,action_size))
        self.gamma = gamma
        self.alpha = alpha
        self.rng = np.random.default_rng()
        self.states = states
        self.actions = actions
    def set_rewards(self, rewards):
        self.rewards = rewards
    def bellman_update(self, s, a, next_s):
        self.table[s,a] = self.table[s,a] + self.alpha * (self.rewards[s,a] + (self.gamma * np.max(self.table[next_s,:])) - self.table[s,a])
    def choose_action(self, s, eps):
        explore = self.rng.random() < eps
        # Cast to a plain Python int: indexing/choice on a numpy array
        # returns numpy.int64, which Pyodide does NOT auto-convert to a JS
        # number when this value later gets passed into a JS function call
        # (e.g. window.drawTransitionArrow) — instead it gets wrapped as an
        # opaque PyProxy, silently breaking any arithmetic/indexing done on
        # it in JS.
        return int(self.rng.choice(self.actions)) if explore else int(self.actions[np.argmax(self.table[s, :])])