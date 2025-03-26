import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        batch_size = x.size(0)
        x_reshaped = torch.reshape(
            x, (batch_size, -1)
        )  # Manipulations to deal with time series format
        output = F.sigmoid(self.fc1(x_reshaped))
        output = self.fc2(output)
        return torch.reshape(
            output, (batch_size, -1, 1)
        )  # Manipulations to deal with time series format
