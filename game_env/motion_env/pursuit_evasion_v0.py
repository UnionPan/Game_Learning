
"""
Petting zoo pursuit evasion environment with single integrator point robot
author: Yunian Pan
"""
import logging
import math
from pettingzoo import ParallelEnv
import numpy as np
from pettingzoo.utils import wrappers
import gym
from gym import spaces
from gym.utils import seeding
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pettingzoo.utils import parallel_to_aec

logger = logging.getLogger(__name__)

MAX_V = 2.0
MIN_V = -2.0
q = 3.0
r_p = 2.0
r_e = 1.7


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
        
        self.v = np.zeros(2)
        
        
    def motion(self, input_u, dt):
    
        # constrain input velocity
        u = np.array(input_u)
        u[0] = np.clip(u[0], self.min_v, self.max_v)
        u[1] = np.clip(u[1], self.min_v, self.max_v)

        # motion model, x_{d+dt} = x_t + u * dt 
        self.state[0] += u[0] * dt
        self.state[1] += u[1] * dt
        self.v = np.copy(input_u)

        return self.state, self.v
        
    def set_init_state(self, init_x, init_y, init_vx=0.0, init_vy=0.0):
        self.state[0] = init_x
        self.state[1] = init_y
        
def plot_arrow(x, y, theta, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(theta), length * math.sin(theta),
              head_length=width, head_width=width)
    plt.plot(x, y)


class PursuitEvasionV0(ParallelEnv):
    metadata = {
        'render_modes': ['human'],
        "name": "Pursuit_Evasion_v0",
        'video.frames_per_second': 2
    }

    def __init__(self, pur_init=None, ev_init=None, obstacles=None, obstacle_radius=None):
        """
        Setting up initial positions of agents and obstacles
        """
        
        self.possible_agents = [ 'pursuer', 'evader']
        self.agents = self.possible_agents[:]
        
        # the maximal velocities for the agents
        self.max_v = MAX_V
        self.min_v = MIN_V
        
        # the radius of the point robot, for calculating the distance to the obstacles
        self.pur_radius = 0.15
        self.ev_radius = 0.15
        
        self.ev_speed = np.zeros(2)
        self.pur_speed = np.zeros(2)
        
        self.dt = 0.1
        
        # initialize the positions, 
        if pur_init:
            self.pur_init_x, self.pur_init_y = pur_init[0], pur_init[1]  
        else:
            self.pur_init_x, self.pur_init_y = np.random.uniform(-10, 10), np.random.uniform(-10, 10)     
        if ev_init:
            self.ev_init_x, self.ev_init_y = ev_init[0], ev_init[1]  
        else:
            self.ev_init_x, self.ev_init_y = np.random.uniform(-10, 10), np.random.uniform(-10, 10) 
        
        
        self.pursuer = PointRobotv0(self.max_v, self.min_v, self.pur_init_x, self.pur_init_y)
        self.evader = PointRobotv0(self.max_v, self.min_v, self.ev_init_x, self.ev_init_y)
         
        
        
        # self.agent_name_mapping = { 'pursuer': self.pursuer, 'evader': self.evader}
        # Define the observation and action spaces
        if obstacles is not None and obstacle_radius is not None:
            self.ob_pos = np.array(obstacles)
            self.ob_radius = np.array(obstacle_radius) # the obstacle radius
        else:
            self.ob_pos = None
            self.ob_radius = None

        self.observation_spaces = {agent: spaces.Box(low=-500, high=500, shape=(8,), dtype=np.float32) for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32) for agent in self.possible_agents}

        
        self.screen_height = 600 # pixel
        self.screen_width = 600 # pixel
        self.state_height = 150 # pixel
        self.state_width = 150 # pixel
        self.world_width = 20.0 # m
        
        self.limit_step = 300
        self.count = 0
        self.action_plot = 0
        # self.percept_region = np.array([-4,4,-2,6])
        # self.basic_sample_reso = 0.1
        # self.sampling_reso_scale = 1.5
        
        self.np_random = None
        self.seed()
        
        self.action_plot = 0

        # record
        self.Q = q * np.eye(2)
        self.R_p = r_p * np.eye(2)
        self.R_e = r_e * np.eye(2)
        
        self.traj_pursuer = []
        self.traj_evader = []
        
        self.collision = False
        self.catch = False
        
        
         
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # def set_obs(self, ob_list, ob_radius):
    #     self.ob_pos = np.array(ob_list)
    #     self.ob_radius = np.array(ob_radius)

    def get_nash(self): 
        # initialize the Nash policy that is calculated from solving the non-zero-sum 
        # LQ game,  reference: Non-cooperative dynamic game theory
        
        K1, K2 = np.eye(2), np.eye(2)
        
        return K1, K2
    
    def get_saddle(self): 
        # initialize the Nash policy that is calculated from solving the zero-sum 
        # LQ game,  reference: Non-cooperative dynamic game theory
        
        K1, K2 = np.eye(2), np.eye(2)
        
        
        return K1, K2
        
    
    def reset(self):
        
        self.agents = np.copy(self.possible_agents) 
        self.ev_init_x, self.ev_init_y = np.random.uniform(-10, 10), np.random.uniform(-10, 10) 
        self.pur_init_x, self.pur_init_y = np.random.uniform(-10, 10), np.random.uniform(-10, 10) 
        
        self.count = 0 
        truncations = None
        self.pursuer.set_init_state(self.pur_init_x, self.pur_init_y)
        self.evader.set_init_state(self.ev_init_x, self.ev_init_y)


        # trajectory
        self.traj_evader = []
        self.traj_evader.append(self.get_eva_state())
        self.traj_pursuer = []
        self.traj_pursuer.append(self.get_pur_state())

        # observations
        observations = { 
               a: (
                   self.get_eva_state(),
                   self.get_eva_speed(),
                   self.get_pur_state(),
                   self.get_pur_speed() 
               )
               for a in self.agents
        }
        
        self.last_obs = observations
        self.last_pur_state = self.get_pur_state()
        self.last_eva_state = self.get_eva_speed()
        
        self.pursuit_strategy = []
        self.traj_pursuer=[]
        self.traj_pursuer.append(self.get_pur_state())
        self.traj_evader=[]
        self.traj_evader.append(self.get_eva_state())
        
        # dummy infos
        infos = {a: {} for a  in self.agents}

        return observations, infos
    
    def step(self, actions):
        self.agents = np.copy(self.possible_agents)
        
        pur_action = actions["pursuer"]
        eva_action = actions["evader"]
        
        # initialize the rewards, termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        truncations = {a: False for a in self.agents}
            # Apply action logic here
            # For example: self.state[agent] = updated_state
        
        infos = {agent: {} for agent in self.agents}
        
        pur_state, pur_speed = self.pursuer.motion(pur_action, self.dt) 
        eva_state, eva_speed = self.evader.motion(eva_action, self.dt) 
        
        self.count += 1
        
        rewards["pursuer"] -=  np.dot(pur_speed.T, self.R_p) @ pur_speed
        rewards["evader"] -= np.dot(eva_speed.T, self.R_e) @ eva_speed
        
        
        if self.robot_collision_with_obstacle(pur_state, self.pur_radius, self.ob_pos, self.ob_radius):
            terminations["pursuer"] = True
            rewards["pursuer"] -= 150
            self.collision = True
            
        if self.robot_collision_with_obstacle(eva_state, self.ev_radius, self.ob_pos, self.ob_radius):
            terminations["evader"] = True
            rewards["evader"] -= 150
            self.collision = True
            
        if np.linalg.norm(self.get_eva_state()[:2]-self.get_pur_state()[:2]) <= self.pur_radius + self.ev_radius:
            terminations = {a: True for a in self.agents }
            rewards["pursuer"] += 200
            rewards["evader"] -= 200
            self.catch = True
        
        if self.count > self.limit_step:
            rewards["pursuer"] += np.dot((pur_state -  eva_state).T, self.Q) @ (pur_state -  eva_state)
            rewards["evader"] -= np.dot((pur_state -  eva_state).T, self.Q) @ (pur_state -  eva_state)
            truncations = {a: True for a in self.agents}

        # calculate the quadratic cost

        observations = { 
               a: (
                   self.get_eva_state(),
                   self.get_eva_speed(),
                   self.get_pur_state(),
                   self.get_pur_speed() 
               )
               for a in self.agents
        }
        
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    

    def robot_collision_with_obstacle(self, x, x_radius, ob_list, ob_radius):
        for i, ob in enumerate(ob_list):
            if np.linalg.norm(x - ob) <= x_radius + ob_radius[i]:
                return True
        
        return False


    def get_pur_state(self):
        state = np.copy(self.pursuer.state)
        return state
    
    def get_pur_speed(self):
        speed = np.copy(self.pursuer.v)
        return speed
    
    def get_eva_state(self):
        state = np.copy(self.evader.state)
        return state
    
    def get_eva_speed(self):
        speed = np.copy(self.evader.v)
        return speed
    
    def why_done(self):
        if self.catch:
            return 0
        elif self.collision:
            return 1
        elif self.count>=self.limit_step:
            return 2
        return False

    def render(self, mode='human'):
        plt.cla()
        x = self.get_pur_state()
        y = self.get_eva_state()
        speed = self.get_pur_speed()
        theta = math.atan2(speed[1], speed[0])
        
        ob = self.ob_pos
        plt.text(-4, -4, str(self.action_plot), fontsize=6)
        plt.plot(x[0], x[1], "xr")
        plt.plot(y[0], y[1], "xb")
        for i, ob in enumerate(self.ob_pos):
            plt.plot(ob[0], ob[1], "ok")
            circle = patches.Circle((ob[0], ob[1]), self.ob_radius[i], edgecolor='red', facecolor='black')
            plt.gca().add_patch(circle)
        plot_arrow(x[0], x[1], theta)

        # # plot laser
        # T_robot = self.get_TF()
        # points = self.scan_to_points(self.last_obs[:self.pursuitor_model.laser_num], self.pursuitor_model.laser_min_angle,
        #                     self.pursuitor_model.laser_increment_angle, T_robot)
        # if len(points)>0:
        #     plt.plot(points[:,0], points[:,1], ".m")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        # print(self.last_obs)
        return np.array([[[1,1,1]]
                         ], dtype=np.uint8)


    def close(self):
        pass

def env():
    env = PursuitEvasionV0()
    env = parallel_to_aec(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# Example usage
if __name__ == "__main__":
    # environment = env()
    # observations = environment.reset()
    # print("Initial observations:", observations)

    # actions = {agent: environment.action_spaces[agent].sample() for agent in environment.agents}
    # observations, rewards, dones, infos = environment.step(actions)
    # print("Observations after step:", observations)
    # print("Rewards:", rewards)
    # print("Dones:", dones)
    # print("Infos:", infos)
    env_f = env()
    
    obstacles = np.array([[0.3, 0.8], [2.4, 5.6]])
    obstacle_radius = np.array([0.5, 1.2])
    
    env = PursuitEvasionV0(obstacles=obstacles, obstacle_radius=obstacle_radius)
    observation, info = env.reset()
    for i in range(300):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.possible_agents}
        actions["pursuer"] = (1/20) * (env.get_eva_state() - env.get_pur_state()) 
        actions["evader"] = np.zeros((2,))
        observation, rewards, terminations, truncations, infos = env.step(actions)
        if any(terminations.values()) or all(truncations.values()):
            observation, infos = env.reset()
        env.render()
        print(rewards)
    