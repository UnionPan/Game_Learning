from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import gymnasium # type: ignore
import functools

class Cournot_v0(ParallelEnv):
    # An m agent Cournot Linear Competition model, usually m = 2, producing n = 1 kinds of goods
    # here, the rewards are the profits made by each single agent given their total production, the market price is a linear model  
    # the states are the first order partial gradients for each agent 
    metadata = {
        "name": "Cournot_Competition_v0"
    }

    def __init__(self, costs, market_capacity, stochastic_option=None):
        """
        input: costs R^(mxn)_+ matrix, each player has an n dimensional cost vector
               market_capacity: R^m_+ vector, the maximum demand for each good
        output: initialize the Nash equilibrium, equilibrium outcome of the game
        """ 
        self._no_agents = len(costs[0]) 
        self._no_goods = len(costs)
        #print(costs)
        print('Initializing {} Firms {} Goods Cournot Competition'.format(self._no_agents, self._no_goods))
        self.market_capacity = market_capacity
        self.costs = costs
        aux_matrix = np.kron(np.ones((self._no_agents, self._no_agents)) + np.eye(self._no_agents), np.eye(self._no_goods))
        vectorized_cost = costs.reshape(-1,1, order='F')
        #print(vectorized_cost)

        aux_capacity = np.tile(market_capacity, (self._no_agents, 1))

        #print('Market Capacity {}'.format(aux_capacity))
        #print(np.shape(aux_matrix), np.shape(vectorized_cost), np.shape(aux_capacity))
        self.eq_strategy = np.linalg.inv(aux_matrix) @ (aux_capacity - vectorized_cost)
        #print(np.shape(self.eq_strategy))
        eq_totalproduction = np.sum(self.eq_strategy.reshape(self._no_goods, self._no_agents), axis=1)
        #print(np.shape(eq_totalproduction))
        self.eq_price = market_capacity - eq_totalproduction.reshape(self._no_goods,1)
        self.stochastic_option = stochastic_option
        if stochastic_option == None:
            self.stochastic_option = { 
                                    'flag': True,
                                    'mean': np.zeros((self._no_goods,1)),
                                    'variance': np.eye(self._no_goods)
                                    }
        self.production = np.zeros((self._no_goods, self._no_agents))
        self.observation = np.zeros((self._no_agents*self._no_goods, 1))
        self.profits = np.zeros((self._no_agents, 1))
        
    def render(self):
        print('*'*32)
        print('*'*5+'Cornout Competition Status'+ '*'*5)
        print(' production ')
        print(self.production)
        print(' partial gradients ')
        print(self.observation)
        print(' profits ')
        print(self.profits)
        print(' Equilibrium Strategies')
        print(self.eq_strategy.reshape(self._no_goods, self._no_agents))
        print(' Equilibrium Prices') 
        print(self.eq_price)
        print('*'*32)
        
    
    def reset(self, seed=None, stochastic_option=None):
        actions = np.array([ self.market_capacity.reshape(self._no_goods,1) / (self._no_agents * 2) for agent_id in range(self._no_agents)])
        self.production = actions
        partial_gradients = np.array([ self.partial_gradient(i, actions) for i in range(self._no_agents)])
        # print(np.shape(partial_gradients))
        # observation = partial_gradients.reshape(-1, 1, order='F')
        self.stochastic_option = stochastic_option
        info = self.stochastic_option
        self.observation =  partial_gradients
        self.profits = [self._cost(i, actions) for i in range(self._no_agents)]
        profits = self.profits
        observation = self.observation
        return observation, profits, info 

    def step(self, actions):
        """
        input: all the agents actions in parallel;
        return the first-order information, i.e., the pseudo gradients 
        also return the zeroth-order information, i.e., the output profit
        """
        self.production = actions
        #print(np.shape(self.production))
        partial_gradients = np.array([ self.partial_gradient(i, actions) for i in range(self._no_agents)])
        # print(np.shape(partial_gradients))
        self.observation = partial_gradients.reshape((self._no_agents, self._no_goods, 1))
        # print(np.shape(self.observation))
        profits = [self._cost(i, actions) for i in range(self._no_agents)]
        info = self.stochastic_option
        observation =  self.observation
        
        return observation, profits, info
    
    def _cost(self, agent_id, actions):
        if agent_id >= self._no_agents:
            raise ValueError("agent ID does not exist!")
        total_production = np.sum(actions, axis=0)
        #print(np.shape(total_production))
        profit = np.dot(np.clip(self.market_capacity -  total_production, a_min=0, a_max=None).T, actions[agent_id])
        return profit 
    
    def partial_gradient(self, agent_id, actions):
        # print(agent_id)
        # print(self.costs[:, agent_id].reshape(self._no_goods,1))
        real_partial_grad = self.market_capacity - actions[agent_id] - np.sum(actions, axis=0) - self.costs[:, agent_id].reshape(self._no_goods,1)
        
        #print(self.stochastic_option is None)
        if self.stochastic_option is not None:
            
            mu = self.stochastic_option['mean']
            sigma = self.stochastic_option['variance']
            xi = np.random.multivariate_normal(mu, sigma).reshape(self._no_goods, 1)
            partial_grad = real_partial_grad + xi   
            #print(xi)         
        else:
            partial_grad = real_partial_grad
        
        return partial_grad
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self._no_goods*self._no_agents,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gymnasium.spaces.Box(low=0.0, high=np.inf, shape=(self._no_goods,), dtype=np.float32)
        
class Cournot_v1(ParallelEnv): 
    metadata = {
        "name": "Cournot_Competition_v0"
    }
    
    def __init__(self, costs): 
        """
        input: costs R^(nxm)_+ matrix, each player has an n dimensional cost vector
               market_capacity: R^n_+ vector, the maximum demand the market has
        initialize the Nash equilibrium, equilibrium outcome of the game
        """ 
        self._no_agents = len(costs) 
        self._no_goods = len(costs[0])
        
        
        
if __name__ == "__main__":
    
    config =  {'flag': True, 'mean': np.zeros((4,1)), 'variance': np.eye(4)}
    P = np.array([[2.0, 4.0, 2.0, 3.0], [3.0, 2.0, 3.5, 1.5], [3.0, 2.5, 3.0, 1.5]]).T
    M = np.array([[18.0], [17.0], [18.0], [19.0]])
    cournot_game = Cournot_v0(costs=P, market_capacity=M, stochastic_option=config)
    cournot_game.render()
    
