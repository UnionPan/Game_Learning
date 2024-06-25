import numpy as np
from cournot_v0 import Cournot_v0
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class CournotAgent:
    def __init__(self, agent_id, initial_production, learning_rate=0.05):
        self.agent_id = agent_id
        self.no_goods = np.shape(initial_production)[0]
        self.learning_rate = learning_rate
        self.production = initial_production
        self.production_trajectory = []
        self.total_production_trajectory = []
        self.price_trajectory = []

    def choose_production(self, observation):
        # The observation should include the partial gradients for all agents
        # Perform gradient ascent
        action = self.production.reshape(observation.shape) + self.learning_rate * observation
        action = np.clip(action, 0, None)  # Ensure action is non-negative
        return action

    def update_production(self, production):
        self.production = production
    
    def log_trajectory(self, production, price):
        total_production = np.sum(production)
        self.total_production_trajectory.append(total_production)
        self.production_trajectory.append(self.production)
        self.price_trajectory.append(price)
    
    def reset_log(self):
        self.total_production_trajectory = []
        self.production_trajectory = []
        self.price_trajectory = []
        
    def plot_trajectory(self, eq_str=None, eq_price=None):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        #print(np.shape(self.production_trajectory))
        for goods_id in range(agent.no_goods):
            plt.plot(np.array(self.production_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.title('No. {} Production Over Time'.format(goods_id + 1))
        plt.xlabel('Time Steps')
        plt.ylabel('Production Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        for goods_id in range(agent.no_goods):
            plt.plot(np.array(self.price_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.title('Market Price Over Time for product No. {}'.format(goods_id + 1))
        plt.xlabel('Time Steps')
        plt.ylabel('Market Price')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
def plot_overall_trajectory(production_trajectories, price_trajectories):
    # calculate the mean and variance
    # print(np.shape(production_trajectories))
    colormap = cm.get_cmap('viridis', np.shape(production_trajectories)[2])
    no_goods = np.shape(production_trajectories)[2]
    mean_trajectory = np.mean(production_trajectories, axis=0)
    # print(np.shape(mean_trajectory))
    variance_trajectory = np.var(production_trajectories, axis=0)
    # print(np.shape(variance_trajectory))
    std_deviation = np.sqrt(variance_trajectory)
    #print(np.shape(mean_trajectory[:, 1] - std_deviation[:, 1]))
    
    time = np.arange(np.shape(production_trajectories)[1])
    mean_trajectory2 = np.mean(price_trajectories, axis=0)
    variance_trajectory2 = np.var(price_trajectories, axis=0)
    std_deviation2 = np.sqrt(variance_trajectory2)
    
    plt.figure(figsize=(6, 4))

    # plt.subplot(1, 2, 1)
        #print(np.shape(self.production_trajectory))
    
    for goods_id in range(no_goods):
        plt.plot(np.array(mean_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        # plt.plot(np.array(actual_trajectory), label=f'product no. {goods_id + 1}')
        plt.fill_between(time, mean_trajectory[:, goods_id, 0] - std_deviation[:, goods_id, 0], mean_trajectory[:, goods_id, 0] + std_deviation[:, goods_id, 0], color=colormap(goods_id), alpha = 0.2)
    plt.title('Production Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Production Curve')
    plt.grid(True)
    plt.legend()

    # plt.subplot(1, 2, 2)
    # for goods_id in range(no_goods):
    #     plt.plot(np.array(mean_trajectory2)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
    #     # plt.plot(np.array(actual_price_trajectory), label=f'product no. {goods_id + 1}')
    #     plt.fill_between(time, mean_trajectory2[:, goods_id, 0] - std_deviation2[:, goods_id, 0], mean_trajectory2[:, goods_id, 0] + std_deviation2[:, goods_id, 0], color=colormap(goods_id), alpha = 0.2)#, label='Variance of product No. {}'.format(goods_id))
    # plt.title('Market Price Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Market Price')
    # plt.legend()

    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    

def plot_tube_trajectory(actual_trajectory, actual_price_trajectory, production_trajectories, price_trajectories):
    # calculate the mean and variance
    # print(np.shape(production_trajectories))
    colormap = cm.get_cmap('viridis', np.shape(production_trajectories)[2])
    no_goods = np.shape(production_trajectories)[2]
    mean_trajectory = np.mean(production_trajectories, axis=0)
    # print(np.shape(mean_trajectory))
    variance_trajectory = np.var(production_trajectories, axis=0)
    # print(np.shape(variance_trajectory))
    std_deviation = np.sqrt(variance_trajectory)
    #print(np.shape(mean_trajectory[:, 1] - std_deviation[:, 1]))
    
    time = np.arange(np.shape(production_trajectories)[1])
    mean_trajectory2 = np.mean(price_trajectories, axis=0)
    variance_trajectory2 = np.var(price_trajectories, axis=0)
    std_deviation2 = np.sqrt(variance_trajectory2)
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
        #print(np.shape(self.production_trajectory))
    
    for goods_id in range(no_goods):
        # plt.plot(np.array(mean_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.plot(np.array(actual_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.fill_between(time, mean_trajectory[:, goods_id, 0] - std_deviation[:, goods_id, 0], mean_trajectory[:, goods_id, 0] + std_deviation[:, goods_id, 0], color=colormap(goods_id), alpha = 0.2)
    plt.title('Production Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Production Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for goods_id in range(no_goods):
        # plt.plot(np.array(mean_trajectory2)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.plot(np.array(actual_price_trajectory)[:, goods_id, 0], label=f'product no. {goods_id + 1}')
        plt.fill_between(time, mean_trajectory2[:, goods_id, 0] - std_deviation2[:, goods_id, 0], mean_trajectory2[:, goods_id, 0] + std_deviation2[:, goods_id, 0], color=colormap(goods_id), alpha = 0.2)#, label='Variance of product No. {}'.format(goods_id))
    plt.title('Market Price Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Market Price')
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
        
        
if __name__ == "__main__":
    # Example of how to use the agent with the environment
    config = {'flag': False, 'mean': np.zeros(4,), 'variance': 0.3 * np.eye(4)}
    
    P = np.array([[2.0, 3.0, 2.0, 3.8], [3.0, 4.0, 3.5, 6.7]]).T
    M = np.array([[8.0], [7.0], [8.0], [12.0]])
    cournot_game = Cournot_v0(costs=P, market_capacity=M, stochastic_option=config)
    x0 = M / 4
    
    agents = [CournotAgent(agent_id=i, initial_production = x0, learning_rate=0.03) for i in range(cournot_game._no_agents)]
    
    # for agent in agents:
    #     agent.update_production(np.zeros((cournot_game._no_goods, 1)))
    
    for agent in agents:
        agent.reset_log()
            
    num_steps = 300
    num_traj = 5
    #print(cournot_game.eq_price)
    
    production_trajectories = []
    price_trajectories = []
    production_trajectories1 = []
    price_trajectories1 = []
    
    
    
    # observations, profits, info = cournot_game.reset()
    # for i in range(num_steps):
    #     actions = []
    #     for agent in agents:
    #         action = agent.choose_production(observations[agent.agent_id])
            
    #         actions.append(action)
    #     observations, profits, info = cournot_game.step(np.array(actions))
        
    #     market_price = M - np.sum(actions, axis=0)
        
    #     for agent, action in zip(agents, actions):
    #         agent.update_production(actions[agent.agent_id])
    #         agent.log_trajectory(actions, market_price)
    
    # actual_traj1, actual_price_traj1 = agents[0].production_trajectory, agents[0].price_trajectory
    # actual_traj2, actual_price_traj2 = agents[1].production_trajectory, agents[1].price_trajectory
    
    
    
    # for agent in agents:
    #     agent.reset_log()
    
    for j in range(num_traj):
        observations, profits, info = cournot_game.reset(config)
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
        production_trajectories.append(agents[0].production_trajectory)
        price_trajectories.append(agents[0].price_trajectory)
        production_trajectories1.append(agents[1].production_trajectory)
        price_trajectories1.append(agents[1].price_trajectory)
        
        for agent in agents:
            agent.reset_log()
    #print(market_price)
    
    plot_overall_trajectory(production_trajectories, price_trajectories)
    plot_overall_trajectory(production_trajectories1, price_trajectories1)

        
    # plot_tube_trajectory(actual_traj1, actual_price_traj1, production_trajectories1, price_trajectories1)
    # plot_tube_trajectory(actual_traj2, actual_price_traj2, production_trajectories2, price_trajectories2)
    