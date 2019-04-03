import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class BanditEnv(gym.Env):
    """
    Description:
        A multi armed bandit environment based on that presented by Sutton and Barto (2018) in 'Reinforcement Learning: An Introduction' used to roughly assess the relative effectiveness of the greedy and epsilon-greedy action-value methods
        
        This testbed randomly generates k-armed bandit problems with the action values selected for each episode according to a normal (Gaussian) distribution with mean 0 and variance 1 and the reward selected at each timestep from a normal distribution with mean as the action value and variance 1.
 
    Source:
        This environment corresponds to the version of the k-armed bandit problem described by Sutton and Barto
    Observation: 
        Type: Box(k)
        Num	Observation               Mean         Variance
        Qt	Action Value Estimate      0.0            1.0
        
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
        self.t = 1 # Zere or One
        self.q_star_mean = 0.0
        self.q_star_var = 1.0
        self.reward_Rt_var = 1.0
        
        self.q_star = None
        self.state_Qt = None
        self.done = None
        
        # Bounds: action value bounds set to -inf to +inf for each action `a`
        state_Qt_high = np.finfo(np.float32).max
        state_Qt_low = np.finfo(np.float32).min

        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(low=state_Qt_low, high=state_Qt_high, shape=(self.k,), dtype=np.float32)


    def reset(self):
        self.q_star = np.random.normal(loc=self.q_star_mean, scale=self.q_star_var, size=self.K)
        self.state_Qt = np.zeros((self.K,)) # initial state (action values) set to zeros
        return np.array(self.state_Qt)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state_Qt = self.state_Qt
        
        reward_Rt = np.random.normal(loc=q_star[action], scale=reward_Rt_var)

        state_Qt[action] = state_Qt[action] + (1/self.t)*(reward_Rt - state_Qt[action])

        self.t += 1
        
        self.state_Qt = state_Qt
        
        return np.array(self.state_Qt), reward_Rt, done, {}

    def render(self, mode='human'):
        print(f'Mean: {self.q_star}')

    def optimal_choice(self):
        return np.argmax(self.q_star)