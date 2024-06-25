
"""
Petting zoo pursuit evasion environment with single integrator point robot
author: Yunian Pan
"""


from pettingzoo import ParallelEnv
import numpy as np
from pettingzoo.utils import wrappers
import gym
from gym import spaces

class PointRobotv0:
    """
    A simple single integrator point robot model
    """
    def __init__(self, max_v, min_v, 
                 init_x, init_y, init_vx=0.0, init_vy=0.0):
        self.state = np.zeros(2) # state = [x,y,yaw,v,w]
        self.set_init_state(init_x, init_y)
        
        self.max_v = max_v
        self.min_v = min_v
        
        
    def motion(self, input_u, dt):
    
        # constrain input velocity
        u = np.array(input_u)
        u[0] = np.clip(u[0], self.min_v, self.max_v)
        u[1] = np.clip(u[1], self.min_v, self.max_v)

        # motion model, x_{d+dt} = x_t + u * dt 
        self.state[0] += u[0] * dt
        self.state[1] += u[1] * dt

        return self.state
        
    def set_init_state(self, init_x, init_y, init_vx=0.0, init_vy=0.0):
        self.state[0] = init_x
        self.state[1] = init_y
        
    
        


class PursuitEvasionV0(ParallelEnv):
    meta_data = {
        'render_modes': ['human'],
        "name": "Pursuit_Evasion_v0"
    }

    def __init__(self):
        """
        A
        """
        self.max_v = 2
        self.min_v = -0.3
        
        self.pur_init_x = 0.0
        self.pur_init_y = 6.0
        
        self.ev_init_x = 10
        self.ev_init_y = 4
        
        self.robot_radius = 0.15
        
        self.target_radius = 0.3
        
        self.dt = 0.1
        
        self.pursuer = PointRobotv0(self.max_v, self.min_v, self.pur_init_x, self.pur_init_y)
        self.evader = PointRobotv0(self.max_v, self.min_v, self.ev_init_x, self.ev_init_y)
         
        
        self.possible_agents = [ 'pursuer', 'evader']
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = { 'pursuer': self.pursuer, 'evader': self.evader}
        # Define the observation and action spaces
        self.ob_pos = np.array([
            [5.0, 6.0],
            [12.0, 12.0]
        ])
        self.ob_radius = np.array([0.7, 0.8] ) # the obstacle radius

        self.observation_spaces = {agent: spaces.Box(low=0, high=255, shape=(5, 5, 3), dtype=np.uint8) for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

        
    def set_obs(self, ob_list, ob_radius):
        self.ob_pos = np.array(ob_list)
        self.ob_radius = np.array(ob_radius)

    
    def reset(self):
        self.agents = self.possible_agents[:]
        self.state = {agent: np.zeros((5, 5, 3), dtype=np.uint8) for agent in self.agents}
        observations = {agent: self.state[agent] for agent in self.agents}
        return observations

    def step(self, actions):
        for agent in self.agents:
            action = actions[agent]
            # Apply action logic here
            # For example: self.state[agent] = updated_state

        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Example logic to set done for all agents
        if some_condition:
            dones = {agent: True for agent in self.agents}

        observations = {agent: self.state[agent] for agent in self.agents if agent in self.agents}
        return observations, rewards, dones, infos

    def render(self, mode="human"):
        for agent in self.agents:
            print(f"{agent} state: {self.state[agent]}")

    def close(self):
        pass

def env():
    env = PursuitEvasionEnv()
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# Example usage
if __name__ == "__main__":
    environment = env()
    observations = environment.reset()
    print("Initial observations:", observations)

    actions = {agent: environment.action_spaces[agent].sample() for agent in environment.agents}
    observations, rewards, dones, infos = environment.step(actions)
    print("Observations after step:", observations)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Infos:", infos)