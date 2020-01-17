# Imports
import numpy as np
import torch
import torch.nn as nn


class Banana_q(nn.Module):
    """
    Banana Q model class.
    
    Attributes
    ----------
    linear_1...4: torch NN linear layer
    
    activation: torch NN activation function
    
    output_modifier: torch NN activation function
    """


    def __init__(self, state_size, action_space_size):
        """
        Initialise Banana Q model class.

        Parameters
        ----------
        state_size: Integer
            Number of dimensions in each state (input size)

        action_space_size: Integer
            Number of dimensions in action space (output size)
        
        Returns
        -------
        None.
        """

        super(Banana_q, self).__init__()

        # Linear layers
        self.linear_1 = nn.Linear(state_size, state_size)
        self.linear_2 = nn.Linear(state_size, state_size)
        self.linear_3 = nn.Linear(state_size, state_size)
        self.linear_4 = nn.Linear(state_size, action_space_size)

        # Non-linearity
        self.activation = nn.LeakyReLU()
        
        # Output modifier
        self.output_modifier = nn.Softmax(dim=1)


    # Full forward pass
    def forward(self, state_batch):
        """ 
        Forward pass through model. 
        
        Parameters
        ----------
        state_batch: torch Tensor (shape: num. batches X state size)
            Batch of all state values to decide actions for
        
        Returns
        -------
        action_probability: torch Tensor
            Softmaxed probabilities for action to take
        """

        actions = self.activation(self.linear_1(state_batch))
        actions = self.activation(self.linear_2(actions))
        actions = self.activation(self.linear_3(actions))
        actions = self.output_modifier(self.linear_4(actions))

        return action_probability   