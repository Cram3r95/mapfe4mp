import torch
from torch import nn
import pdb

from model.modules.layers import MLP

class Classifier(nn.Module):

    def __init__(self, mlp_config):
        super(Classifier, self).__init__()

        self.spatial_embedding = MLP(**mlp_config) 
        # self.softmax = nn.Softmax(dim=1) # Perform softmax operation accross hidden_dim axis
        self.sigmoid = nn.Sigmoid()
    """
    # Sotfmax: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    def forward(self, trajectory_predicted, trajectory_groundtruth):
        embedding = [trajectory_predicted,trajectory_groundtruth]
        return self.softmax(embedding) # Return binary classification: Fake (Predicted) vs True (Groundtruth) trajectories
    """

    def forward(self, input_data):
        """
        Inputs:
        - input_data: Tensor of shape (self.num_layers (LSTM encoder discriminator), batch, self.hidden_dim)
        Output:
        - predicted_labels: Predicted label from the discriminator (Batch size x 2)
        """

        input_softmax = self.spatial_embedding(
            input_data
        )
        # pdb.set_trace()
        # print("input_softmax ", input_softmax)

        scores = self.sigmoid(input_softmax)

        return scores