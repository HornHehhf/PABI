import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.log_softmax(out)
