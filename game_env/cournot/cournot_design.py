from cournot_agent import CournotAgent, plot_overall_trajectory, plot_tube_trajectory
from cournot_v0 import Cournot_v0
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class CournotDesigner:
    def __init__(self, Cournot_Game):
        self.design_variable = 0.01 * np.ones(Cournot_Game._no_agents)
        
    def designer_objective(self):
        pass
        
    def designer_iter(self):
        pass
    

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
    
    