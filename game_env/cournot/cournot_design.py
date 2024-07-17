from cournot_agent import CournotAgent, plot_overall_trajectory, plot_tube_trajectory
from cournot_v0 import Cournot_v0
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

class CournotDesigner:
    # CournotDesigner class, design learning parameters for two firms to make one firm win during the transient learning process
    # initializing: theta, the game, initial distribution, agents 
    def __init__(self, Cournot_Game, config):
        self.design_variable = 0.01 * np.ones(Cournot_Game._no_agents)
        
        self.game = Cournot_Game
        self.initial = [ np.clip( np.random.multivariate_normal(self.game.market_capacity.reshape(self.game._no_goods,) / (2.5 * self.game._no_agents), 1e-10 * np.eye(self.game._no_goods)).reshape(self.game._no_goods, 1), a_min=0, a_max=None) for _ in range(self.game._no_agents)]
    
        self.config = config
        self.agents = [CournotAgent(agent_id=i, initial_production = self.initial[i], learning_rate=self.design_variable[i]) for i in range(self.game._no_agents)]
    
        
    def zeroth_estimate(self, num_traj = 25, num_steps = 50, radius = 0.01 ):
        # estimating the searching direction of the design parameter using zero-th order
        # every iteration we collect num_traj trajectories and averaging their 
        # for i in range(num_traj):
        prod_traj1, prod_traj2 = [], []
        price_traj1, price_traj2 = [], []
        profit_traj1, profit_traj2 = [], []
        
        lambdaH = np.zeros(self.design_variable.shape)    # initialize the gradient estimate
        
        for j in range(num_traj):
            # reset the game
            profit1, profit2 = [], []
            observations, profits, info = self.game.reset(initial=self.initial, stochastic_option=self.config)
            #  sample a learning hyperparameter theta_m = theta + u_m
            perturbation = np.random.normal(0,1,len(self.design_variable)).reshape(self.design_variable.shape)
            perturbation = ( radius / np.linalg.norm(perturbation) ) * perturbation
            print(perturbation)
            
            perturbed_lr = self.design_variable + perturbation
            for agent in range(len(self.agents)):
                self.agents[agent].learning_rate = perturbed_lr[agent]
            for i in range(num_steps):
                actions = []
                for agent in self.agents:
                    action = agent.choose_production(observations[agent.agent_id])
            
                    actions.append(action)
        
                observations, profits, info = self.game.step(np.array(actions))
                
                profit1.append(profits[0].tolist())
                profit2.append(profits[1].tolist())
            
                market_price = self.game.market_capacity - np.sum(actions, axis=0)
        
                for agent, action in zip(self.agents, actions):
                    agent.update_production(actions[agent.agent_id])
                    agent.log_trajectory(actions, market_price)
        
            #cournot_game.render()
            prod_traj1.append(self.agents[0].production_trajectory)
            prod_traj2.append(self.agents[1].production_trajectory)
            price_traj1.append(self.agents[0].price_trajectory)
            price_traj2.append(self.agents[1].price_trajectory)
            profit_traj1.append(profit1)
            profit_traj2.append(profit2)
        
            for agent in self.agents:
                agent.reset_log()
            
            # estimating the trajectory reward for the designer
            #print(np.array(profit1).shape)
            
            Hm = self.design_objective(np.array(profit1), np.array(profit2))
            
            #  the gradient estimate is using (1/M) * sum_m () H_m * U_m  + scaling * (nabla regularizer)
            lambdaH += ( 1 / (num_steps * num_traj ) ) * Hm * perturbation
        lambdaH += 0.01 * (self.design_variable[0].tolist()  - self.design_variable[1].tolist()) * np.array([[1], [-1]]).reshape(self.design_variable.shape) 
        #print(lambdaH)
        #print(np.average(np.average(profit_traj1, axis=0).reshape(-1) - np.average(profit_traj2, axis=0).reshape(-1) ))
            
        return prod_traj1, prod_traj2, price_traj1, price_traj2, profit_traj1, profit_traj2,  lambdaH
    

    def design_objective(self, profit_traj1, profit_traj2):
        res = np.average(profit_traj1 - profit_traj2).tolist()
        #print(res)
        return res
        
            

    def design_iter(self, num_designiter=50, search_rad=0.01, plot=True):
        # essentially this evolutionary hyperparameter optimization
        # with the searching direction to be a certain criteria that firm 1 
        # has to win 
    
        # for i in tdqm(range(K)):
        #     for j in range():
        #         traj = self.
        #         lambdaH = self.designer_iter(traj)
        
        # intializing the data
        market_share = []
        market_share_var = []
        
        profit_diff = []
        profit_diff_var = []
        
        avg_market_price = []
        avg_market_price_var = []
        
        for i in tqdm(range(num_designiter)):
            prod_traj1, prod_traj2, price_traj1, price_traj2, profit_traj1, profit_traj2,  lambdaH = self.zeroth_estimate(radius=search_rad)
            self.design_variable =  np.clip( self.design_variable + lambdaH , a_min=0.001, a_max=0.5 )
            print(self.design_variable)
            
            profit_traj1, profit_traj2  = np.squeeze(np.array(profit_traj1)), np.squeeze(np.array(profit_traj2))
            prod_traj1, prod_traj2 = np.squeeze(np.array(prod_traj1)), np.squeeze(np.array(prod_traj2))
            price_traj1, price_traj2 = np.squeeze(np.array(price_traj1)), np.squeeze(np.array(price_traj2)) 
             
            # print(price_traj1.shape, prod_traj1.shape, profit_traj1.shape) 
            profit_diff.append(np.mean(profit_traj1 - profit_traj2, axis=( 0, 1)))
            profit_diff_var.append(np.sqrt(np.var(profit_traj1 - profit_traj2, axis=(0, 1))))

            avg_market_price.append(np.mean(price_traj1, axis=( 0, 1)))
            avg_market_price_var.append(np.sqrt(np.var(price_traj1, axis = (0,1))))
            
            
            market_share.append(np.mean( prod_traj1/(prod_traj1 + prod_traj2), axis = (0,1) ))
            market_share_var.append(np.sqrt(np.var(prod_traj1/(prod_traj1 + prod_traj2), axis = (0,1))))
            
        # print(np.array(avg_market_price).shape)
        # print(np.array(profit_diff_var).shape)
        # print(np.array(market_share_var).shape)
        # print(np.array(profit_traj1).shape)
            
            
        if plot:
            avg_market_price, avg_market_price_var = np.array(avg_market_price),  np.array(avg_market_price)
            profit_diff, profit_diff_var = np.array(profit_diff), np.array(profit_diff_var)
            market_share, market_share_var = np.array(market_share), np.array(market_share_var)
            
            
            #print(np.shape(self.production_trajectory))
            time = np.arange(profit_diff.shape[0])
            plt.plot(profit_diff)
        # plt.plot(np.array(actual_trajectory), label=f'product no. {goods_id + 1}')
            plt.fill_between(time, profit_diff - profit_diff_var, profit_diff + profit_diff_var, color='purple', alpha = 0.2)

            plt.title('Average Profit Dominance ')
            plt.xlabel('Designer Iter')
            plt.ylabel('Production Curve')
            plt.grid(True)
            plt.show()
            
            colormap = cm.get_cmap('viridis', np.shape(avg_market_price)[1])
            
            for goods_id in range(self.game._no_goods):
                plt.plot(np.array(market_share)[:, goods_id], label=f'product no. {goods_id + 1}')
                plt.fill_between(time, market_share[:, goods_id] - market_share_var[:, goods_id], market_share[:, goods_id] + market_share_var[:, goods_id], color=colormap(goods_id), alpha = 0.2)
            plt.title('Market share Curve')
            plt.xlabel('Designer Iteration')
            plt.ylabel('Market share (%)')
            plt.grid(True)
            plt.legend()
            plt.show()
            # Plotting the market share of firm 1 evolution for different goods: 
            # for goods_id in range(agent.no_goods):
            #     plt.plot(np.array(self.price_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
            # plt.title('Market Price Over Time for product No. {}'.format(goods_id + 1))
            # plt.xlabel('Designer Iter')
            # plt.ylabel('Market Price')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()
        
            
            
            
        
    

if __name__ == "__main__":
    
    config = {'flag': False, 'mean': np.zeros(4,), 'variance': 0.5 * np.eye(4)}
    
    P = np.array([[2.0, 3.0, 2.0, 3.8], [3.0, 4.0, 3.5, 6.7]]).T
    M = np.array([[8.0], [7.0], [8.0], [12.0]])
    
    cournot_game = Cournot_v0(costs=P, market_capacity=M, stochastic_option=config)
    x0 = M / 4
    
    designer = CournotDesigner(cournot_game)
    designer.design_variable[0] = 0.5
    designer.design_variable[1] = 0.05
    agents = [CournotAgent(agent_id=i, initial_production = x0, learning_rate=designer.design_variable[i]) for i in range(cournot_game._no_agents)]
    
    num_steps = 300
    num_traj = 5
    #print(cournot_game.eq_price)
    production_trajectories1 = []
    production_trajectories2 = []
    price_trajectories1 = []
    price_trajectories2 = []
    
    
    
    observations, profits, info = cournot_game.reset()
    for i in range(num_steps):
        actions = []
        for agent in agents:
            action = agent.choose_production(observations[agent.agent_id])
            
            actions.append(action)
        observations, profits, info = cournot_game.step(np.array(actions))
        
        market_price = M - np.sum(actions, axis=0)
        
        for agent, action in zip(agents, actions):
            agent.update_production(actions[agent.agent_id])
            agent.log_trajectory(actions, market_price)
    
    actual_traj1, actual_price_traj1 = agents[0].production_trajectory, agents[0].price_trajectory
    actual_traj2, actual_price_traj2 = agents[1].production_trajectory, agents[1].price_trajectory
    
    
    
    for agent in agents:
        agent.reset_log()
    
    
    for j in range(num_traj):
        observations, profits, info = cournot_game.reset(stochastic_option= config)
        for i in range(num_steps):
            actions = []
            for agent in agents:
                action = agent.choose_production(observations[agent.agent_id])
            
                actions.append(action)
        
            observations, profits, info = cournot_game.step(np.array(actions))
        
            market_price = M - np.sum(actions, axis=0)
        
            for agent, action in zip(agents, actions):
                agent.update_production(actions[agent.agent_id])
                agent.log_trajectory(actions, market_price)
        
            #cournot_game.render()
        production_trajectories1.append(agents[0].production_trajectory)
        production_trajectories2.append(agents[1].production_trajectory)
        price_trajectories1.append(agents[0].price_trajectory)
        price_trajectories2.append(agents[1].price_trajectory)
        
        for agent in agents:
            agent.reset_log()
    #print(market_price)
    
    plot_overall_trajectory(production_trajectories1, price_trajectories1)
    plot_overall_trajectory(production_trajectories2, price_trajectories2)
    
    plot_tube_trajectory(actual_traj1, actual_price_traj1, production_trajectories1, price_trajectories1)
    plot_tube_trajectory(actual_traj2, actual_price_traj2, production_trajectories2, price_trajectories2)
    
    