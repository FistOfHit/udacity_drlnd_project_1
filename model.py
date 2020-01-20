# Imports
import torch.nn as nn


class Q_network(nn.Module):
    """
    Q-Network model class.
    
    Attributes
    ----------
    linear_1, 2 ,3: torch NN linear layer
    
    activation: torch NN activation function
    
    output_modifier: torch NN activation function
    """


    def __init__(self, state_size, action_space_size):
        """
        Initialise Q-Network model class.

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

        super(Q_network, self).__init__()

        # Linear layers
        self.linear_1 = nn.Linear(state_size, 3*state_size)
        self.linear_2 = nn.Linear(3*state_size, 2*state_size)
        self.linear_3 = nn.Linear(2*state_size, action_space_size)

        # Non-linearity
        self.activation = nn.ReLU()


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
        action_probs: torch Tensor
            Softmaxed probabilities for action to take
        """

        action_probs = self.activation(self.linear_1(state_batch))
        action_probs = self.activation(self.linear_2(action_probs))
        action_probs = self.linear_3(action_probs)

        return action_probs    