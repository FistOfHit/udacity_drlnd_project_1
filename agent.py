import experience_replay as exp_rep
import model
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# GPU check
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'


class Agent():
    """
    Agent class to interact with environment and learns a policy.
    
    Attributes
    ----------
    q_current: Q-Network object
        Current version of Q-Network to use as policy   
        
    q_target: Q-Network object
        Target version of Q-Network to use to update current policy
        
    memory: Replay_buffer object
        Experience replay buffer to store and sample experience tuples
        
    optimiser: Torch Optim object
        Optimisation algorithm to update Q-Network weights
        
    num_time_steps: Integer
        Number of time steps passed since last update
    """

    def __init__(self, state_size, action_space_size,
                 nn_hyperparams=[5e-4, 64, 4],
                 rl_hyperparams=[int(1e5), 0.99, 1e-3, True]):
        """
        Initialize an Agent.
        
        Parameters
        ----------
        state_size: Integer
            Number of dimensions in any state
            
        action_space_size: Integer 
            Number of all possible actions available to agent in any state
            
        nn_hyperparams: List
            List of all hyper-parameters for neural network. Consists of, in
            this order:
                
            learn_rate: Float (default: 5e-4)
                (Initial) Learning rate for network optimiser
                
            batch_size: Integer (default: 64)
                How many (S, A, R, S') tuples in one batch
                
            update_freq: Integer (default: 4)
                Per how many iterations is network updated
        
        rl_hyperparams: List
            List of all hyper-parameters for the RL problem. Consists of, in
            this order:
                
            buffer_size: Integer (default: 1e5)
                Number of (S, A, R, S') tuples to store in memory for replay
                
            gamma: Float (default: 0.99)
                Discount rate applied to future rewards
                
            tau: Float (default: 1e-3)
                Proportion to use when updating target network
                
            prioritised: Bool (default: True)
                Whether or not to use prioritised experience replay
        
        Returns
        -------
        None.
        """
        
        # Unpack parameters
        self.learn_rate, self.batch_size, self.update_freq = nn_hyperparams
        self.buffer_size, self.gamma, self.tau, self.prioritised = rl_hyperparams
        
        # Initialise enviroment parameters
        self.state_size = state_size
        self.action_space_size = action_space_size

        # Initialise Q-Network (policy)
        self.q_current = model.Q_network(state_size, action_space_size).to(device)
        self.q_target = model.Q_network(state_size, action_space_size).to(device)
        
        # Initialise optimiser with defined learning rate
        self.optimiser = optim.Adam(self.q_current.parameters(), 
                                    lr=self.learn_rate)

        # Initialise experience replay buffer
        self.memory = exp_rep.Replay_buffer(action_space_size, self.buffer_size,
                                            self.batch_size, self.prioritised)
        
        # Initialize time step coutner (for updating every update_freq steps)
        self.num_time_steps = 0
        
        return
    
    
    def step(self, state, action, reward, next_state, done):
        """
        Carry out one time step.
        
        Parameters
        ----------
        state: Torch Tensor
            Tensor of one state at start of timestep
            
        action: Integer
            Action taken in state
            
        reward: Float
            Reward gained by taking action in state
            
        next_state: torch Tensor
            Next state achieved by taking action in state
            
        done: Bool
            TO FILL
            
        Returns
        -------
        None.
        """
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn once every update_frequency time steps.
        self.num_time_steps = (self.num_time_steps + 1) % self.update_freq
        
        # If we've had update_freq timesteps since last update, then update
        if self.num_time_steps == 0:
            
            # Only if we have enough memory
            if len(self.memory) > self.batch_size:
                
                # Draw sample from memory
                experiences = self.memory.randomly_sample()
                
                # Learn with these sampled experiences from memory
                self.learn(experiences, self.gamma)
        
        return


    def act(self, state, epsilon=0):
        """
        Provide an action for the agent to perform with a given state.
        
        Parameters
        ----------
        state: Torch Tensor
        Current state of agent
                
        epsilon: Float (default: 0)
            epsilon value, for epsilon-greedy action selection
            
        Returns
        -------
        chosen_action: Integer
            The action chosen based on the policy
        """
        
        # Prepare state for Q-Network
        state = state.float().unsqueeze(0).to(device)
        
        # Set Q-Network to evaluation mode
        self.q_current.eval()
        
        # Get action
        action_probs = self.q_current.forward(state)
        self.q_current.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            chosen_action = np.argmax(action_probs.cpu().data.numpy())
        else:
            chosen_action = random.choice(np.arange(self.action_size))
            
        return chosen_action
        

    def learn(self, experiences, gamma):
        """
        Update action-value function (Q-Network) parameters.

        Parameters
        ----------
        experiences: Tuple (shape: 5 X batch_size)
            tuple of (s, a, r, s', done) tuples sampled from memory
        gamma: Float
            Discount factor to be applied to future rewards
            
        Returns
        -------
        None.
        """
        
        # Unpack components from tuple
        states, actions, rewards, next_states, dones = experiences
        
        # Set Q-Network to train mode
        self.q_current.train()

        # Get max predicted Q values (for next states) from target model
        next_q_targets = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        current_q_targets = rewards + (gamma * next_q_targets * (1 - dones))

        # Get expected Q values from local model
        expected_q_values = self.q_current(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss(expected_q_values, current_q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target Q-Network
        self.soft_update(self.q_current, self.q_target, self.tau)   
        
        return
                  

    def soft_update(self, current_model, target_model, tau):
        """
        Soft update model parameters.

        Parameters
        ----------
        local_model: Q_network object
            Network that weights will be updated from
            
        target_model: Q_network object
            Network that weights will be updated to
        
        tau: Float
            Interpolation parameter for update
        
        Returns
        -------
        None.
        """
        
        for target_param, local_param in zip(target_model.parameters(),
                                             current_model.parameters()):
            target_param.data.copy_(tau*local_param.data + \
                                    (1 - tau)*target_param.data)
            
        return
