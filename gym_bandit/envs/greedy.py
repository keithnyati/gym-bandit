'''Multi Armed Bandit - KTM'''

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np

class GreedyBanditEnv(gym.Env):
    """
    Description:
        A multi armed bandit environment based on that presented by Sutton and Barto (2018) in 'Reinforcement Learning: An Introduction' used to roughly assess the relative effectiveness of the greedy and epsilon-greedy action-value methods
        
        This testbed randomly generates k-armed bandit problems with the action values selected for each episode according to a normal (Gaussian) distribution with mean 0 and variance 1 and the reward selected at each timestep from a normal distribution with mean as the action value and variance 1.
 
    Source:
        This environment corresponds to the version of the k-armed bandit problem described by Sutton and Barto
    Observation: 
        Type: Box(1 + 2*k)
        Sym ix	    Observation             Min  Max   Mean  Variance
        t   [0]     Timestep                0    1000
        n   [1:k+1] Times Action Selected   0    1000
        Qn	[k+1:]  Action Value Estimate  -inf  +inf  0.0   1.0
    Actions:
        Type: Discrete(k)
        Num	Action
        a	Pull arm a
        
        Note: The action value estimate, Qt, for each action,a, is reduced or increased and is not fixed; it depends on both the true action values (hidden) and the observed rewards per action.
    Reward:
        Reward, Rt, is selected from a normal distribution with mean `q_star` and variance 1.
    Starting State:
        All observations are assigned a starting value of 0.
    Episode Termination:
        Run out of timesteps
        Solved Requirements
        Considered solved when the average reward is greater than or equal to the theoretical maximum of 1.55?
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.k = 10
        self.T = 1000 # Number of timesteps to end of episode
        self.q_star_mean = 0.0
        self.q_star_var = 1.0
        self.Rt_var = 1.0
        
        # Bounds: action value bounds set to -inf to +inf for each action `a`
        Qn_high = np.finfo(np.float32).max
        Qn_low = np.finfo(np.float32).min
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(low=Qn_low, high=Qn_high, shape=(1 + 2 * self.k,), dtype=np.float32)

        self.q_star = None
        self.state = None
        self.done = None
        self.steps_beyond_done = None

    def reset(self):
        # 'true' action value for each arbitrary action `a`
        self.q_star = np.random.normal(loc=self.q_star_mean, scale=self.q_star_var, size=self.k)

        # init state: columns = timestep (t), times action selected (n), estimate action value (Qn)
        self.state = np.zeros(1 + 2 * self.k,)

        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        At = action

        state = self.state
        t, n, Qn = state[0], state[1:self.k+1], state[self.k+1:]
        
        # update timestep
        t += 1

        # Rt (reward) selected per timestep according to a normal (Gaussian) ...
        # ... distribution with mean as q_star (true action value) and variance 1
        Rt = np.random.normal(loc=self.q_star[At], scale=self.Rt_var)
        
        # update number of times action (At) has been selected
        n[At] += 1

        # update estimated action value for selected action (At = a)
        Qn[At] = Qn[At] + (1/n[At])*(Rt - Qn[At])
        
        # termination state is the end of timesteps (T)
        done = bool(t < self.T)

        if not done:
            reward = Rt
        elif self.steps_beyond_done is None:
            # End of episode!
            self.steps_beyond_done = 0
            reward = Rt
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        state[0], state[1:self.k+1], state[self.k+1:] = t, n, Qn 
        
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        print(f'Mean: {self.q_star}')

    def sample(self):
        a = np.random.randint(0, self.k)
        return self.step(a)