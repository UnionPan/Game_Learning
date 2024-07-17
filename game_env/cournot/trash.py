from cournot_agent import CournotAgent, plot_overall_trajectory, plot_tube_trajectory
from cournot_v0 import Cournot_v0
from cournot_design import CournotDesigner
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm


    
P = np.array([[4.1, 4.0, 3.3], [4.1, 4.0, 3.3]]).T
M = np.array([[28.0], [17.0], [25.0]])

config = {'flag': False, 'mean': np.zeros(P.shape[0],), 'variance': 0.5 * np.eye(P.shape[0])}
    
cournot_game = Cournot_v0(costs=P, market_capacity=M, stochastic_option=config)
designer = CournotDesigner(cournot_game, config)

#print(designer.agents, designer.initial)
#print(designer.game.eq_price, designer.game.eq_strategy)
# designer.game.reset(designer.initial, stochastic_option=config)

designer.design_iter(num_designiter=30)