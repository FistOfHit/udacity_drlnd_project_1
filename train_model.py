# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch


def deep_q_learning(agent, env, brain_name,
                    q_hyperparameters=[2000, 1000, 1, 0.01, 0.995]):
    """
    Perform Deep Q-Learning with a given agent and hyperparameters.
    
    Parameters
    ----------
    agent: Agent object
        Agent to train with a given environment
        
    env: Unity environment object
        Environment to train an agent to navigate
        
    brain_name: String
        Name of the "brain" being used currently
        
    q_hyperparameters: List
        List of Q-Learning hyperparameters, containing in this order:
            
        num_episodes: Integer
            Maximum number of training episodes to perform
        
        max_timesteps: Integer
            Maximum number of timesteps to perform per episode
            
        epsilon_start: Float
            Initial value of epsilon, for epsilon-greedy action selection
            
        epsilon_end: Float
            Minimum possible value of epsilon
            
        epsilon_decay: Float
            Multiplicative decay rate (per episode) for decreasing epsilon
            
    Returns
    -------
    scores: List
        List of score achieved by agent in each episode
    """
    
    # Unpack Q-Learning parameters
    num_episodes, max_timesteps, epsilon_start, epsilon_end, \
        epsilon_decay = q_hyperparameters
    
    # Tracking scores and initialise epsilon
    scores = []
    epsilon = epsilon_start
    
    # Iterate through all episodes
    for episode in range(num_episodes):
        
        # Reset state and cumulative reward
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.Tensor(env_info.vector_observations[0])
        score = 0
        
        # Iterate until the end of the episode
        for time in range(max_timesteps):
            
            # Agent choses an action based on policy
            action = agent.act(state, epsilon)

            # Environment observes agent acting, provides new state and reward
            env_info = env.step(action)[brain_name]
            next_state = torch.Tensor(env_info.vector_observations[0])
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            # Agent acts and possibly learns from experiences, stores in memory
            agent.step(state, action, reward, next_state, done)
            
            # Update state and cumulative reward
            state = next_state
            score += reward
            
            # If episode is complete
            if done:
                break 
        
        # Update episodic parameters
        scores.append(score)
        epsilon = max(epsilon_end, epsilon_decay*epsilon)
        running_average = np.mean(scores[-100:])
        
        # Print update
        print('Episode: %d, Average Score: %.2f' %\
              (episode, running_average))
        
        # If agent has reached a high enough score on average, save policy
        if running_average >= 200:
            print('\nEnvironment solved in %d episodes! \n' +
                  'Average Score: %.2f' % (episode-50, running_average))
            torch.save(agent.q_current.state_dict(), 'checkpoint.pth')
            break
        
    return scores


def plot_scores(scores):
    """
    Generate a plot of the scores from the agent as it learned.
    
    Parameters
    ----------
    scores: List
        List of scores after each episode agent went through
        
    Returns
    -------
    None.
    """

    # Initialise figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot raw scores
    x_axis = np.arange(len(scores))
    plt.plot(x_axis, scores, 'b')
    
    # Calculate 100 episode running mean
    running_mean = np.convolve(scores, np.ones(100), 'valid') / 100
    x_axis = np.arange(len(running_mean))
    plt.plot(x_axis, running_mean, 'r')
    
    plt.legend(["Score", "100-Episode running mean"])
    
    # Plot 100 episode moving average
    plt.ylabel('Score achieved')
    plt.xlabel('Episodes passed')
    plt.title('Score achieved by agent agaist episodes experienced')
    
    plt.show()

    return


