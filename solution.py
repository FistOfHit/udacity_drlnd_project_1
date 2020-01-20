# Imports
import agent 
from unityagents import UnityEnvironment
import train_model as tm

env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset environment
env_info = env.reset(train_mode=True)[brain_name]

# Size of task (parameters we need)
state = env_info.vector_observations[0]
state_size = len(state)
action_space_size = brain.vector_action_space_size

# Initialise an agent
agent = agent.Agent(state_size, action_space_size)

# Train agent with environment and rewards
scores = tm.deep_q_learning(agent, env, brain_name)

# Plot scores
tm.plot_scores(scores)

# Properly close environment
env.close()