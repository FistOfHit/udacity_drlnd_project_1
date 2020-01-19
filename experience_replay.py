import collections as cs
import numpy as np
import random
import torch


# GPU check
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'
    

class Replay_buffer:
    """
    Replay buffer to store and provide experience tuples.
    
    Attributes
    ----------
    memory: Double-ended Queue object
        Constant length, modifiable memory list
        
    experience: Named Tuple object
        Tuple object to store sample of experiences
    """

    def __init__(self, action_space_size, buffer_size, batch_size, prioritised):
        """
        Initialize a Replay_buffer object.

        Parameters
        ----------
        action_space_size: Integer
            Number of possible actions available in any state
            
        buffer_size: Integer
            Maximum size of experience buffer
            
        batch_size: Integer
            Size of each training batch
            
        prioritised: Bool
            Whether or not to prioritise experience replay
            
        Returns
        -------
        None.
        """
        
        # Initialise parameters
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        
        # Initialise experience storage
        self.memory = cs.deque(maxlen=buffer_size)  
        self.experience = cs.namedtuple("Experience",
                                        field_names=["state", "action",
                                                     "reward", "next_state",
                                                     "done"])
        
        return
    
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience tuple to memory.
        
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
        
        # Append new experience
        new_experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_experience)
        
        return
    
    
    def randomly_sample(self):
        """
        Randomly sample a batch of experiences from memory.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        sampled_experiences: Tuple (shape: 5 X batch_size)
            (s, a, r, s') tuples for experience replay
        """
        
        # Randomly select batch_size experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

        # Unpack
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences \
                                             if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if \
                                              exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences \
                                              if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if \
                                                  exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences \
                                            if exp is not None]).astype(np.uint8)).float().to(device)
        
        # Pack into another tuple (transpose effectivley)
        sampled_experiences = (states, actions, rewards, next_states, dones)
  
        return sampled_experiences
    
    
    def prioritised_sample(self):
        """
        Prioritised sample a batch of experiences from memory.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        sampled_experiences: Tuple (shape: 5 X batch_size)
            (s, a, r, s') tuples for experience replay
        """
        
        # Randomly select batch_size experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

        # Unpack
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences \
                                             if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if \
                                              exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences \
                                              if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if \
                                                  exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences \
                                            if exp is not None]).astype(np.uint8)).float().to(device)
        
        # Pack into another tuple (transpose effectivley)
        sampled_experiences = (states, actions, rewards, next_states, dones)
  
        return sampled_experiences


    def __len__(self):
        """
        Return the current size of internal memory.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        length: Integer
            Length of memory deque
        """
        
        length = len(self.memory)
        
        return length
