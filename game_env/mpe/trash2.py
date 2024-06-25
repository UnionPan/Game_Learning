import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils.conversions import parallel_wrapper_fn




# Reset the environment to start
observations = env.reset()

# Collect trajectory data
agent_positions = {agent: [] for agent in env.agents}
agent_index = {agent: idx for idx, agent in enumerate(env.agents)}

# Run the environment for a fixed number of steps
num_steps = 500

for step in range(num_steps):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Render the environment
    env.render()
    
    # Store positions of each agent
    for agent in env.agents:
        if not terminations[agent] and not truncations[agent]:
            idx = agent_index[agent]
            agent_positions[agent].append(env.world.agents[idx].state.p_pos)

# Convert positions to numpy arrays for easier manipulation
for agent in agent_positions:
    agent_positions[agent] = np.array(agent_positions[agent])

# Extract landmark positions
landmark_positions = [entity.state.p_pos for entity in env.world.landmarks]

# Plot the trajectory
plt.figure(figsize=(10, 10))
for agent, positions in agent_positions.items():
    if positions.size > 0:  # ensure there are positions to plot
        plt.plot(positions[:, 0], positions[:, 1], label=agent)

# Plot landmarks
for landmark_pos in landmark_positions:
    plt.scatter(landmark_pos[0], landmark_pos[1], s=100, c='red', marker='X', label='landmark')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Trajectories in simple_tag Environment with Landmarks')
plt.legend()
plt.grid()
plt.show()

# Close the environment
env.close()

