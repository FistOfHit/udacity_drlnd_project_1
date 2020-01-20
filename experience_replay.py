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

    def __init__(self, action_space_size, buffer_size, batch_size):
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
                                                     "done", "priority"])
        
        return
    
    
    def add(self, state, action, reward, next_state, done, priority):
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
            Whether or not this state is when the episode is done
            
        priority: Float
            Priority value given to experience
            
        Returns
        -------
        None.
        """
        
        # Append new experience
        new_experience = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(new_experience)
        
        return
    
    
    def sample(self):
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
        
        # Get the priorities
        priorities = torch.Tensor([exp.priority for exp in self.memory if exp is not None])
        probability = (priorities**0.5 / torch.sum(priorities**0.5)).numpy()
        
        # Randomly sample some indexes with given sampling probabilities
        indexes = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=probability)
        indexes = np.array(indexes, dtype=np.int)
        
        experiences = []
        for index in indexes:
            experiences.append(self.memory[index])

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
        priorities = torch.from_numpy(np.vstack([exp.priority for exp in experiences \
                                            if exp is not None])).float().to(device)
        
        # Pack into another tuple (transpose effectivley)
        sampled_experiences = (states, actions, rewards, next_states, dones, priorities)
  
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
