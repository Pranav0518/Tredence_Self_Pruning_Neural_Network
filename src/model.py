import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# PRUNABLE LINEAR
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))  
        self.bias = nn.Parameter(torch.zeros(out_features))  
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))  

        nn.init.kaiming_normal_(self.weight)  
        nn.init.constant_(self.gate_scores, 2.0)  # initialize gates near 1 after sigmoid

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)  # convert gate scores to [0,1]
        return F.linear(x, self.weight * gates, self.bias)  # apply gated weights


# ==============================
# RESIDUAL BLOCK
# ==============================
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)  # first convolution layer
        self.bn1 = nn.BatchNorm2d(c)  # normalize activations
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)  # second convolution layer
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        # residual connection improves gradient flow and stability
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


# ==============================
# MODEL
# ==============================
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # extract low-level features
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),  # improve feature learning
            nn.MaxPool2d(2),  # reduce spatial size

            nn.Conv2d(32, 64, 3, padding=1),  # deeper feature extraction
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2)
        )

        self.fc1 = PrunableLinear(64 * 8 * 8, 512)  # prunable dense layer
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = PrunableLinear(512, 10)  # output layer for 10 classes

    def forward(self, x):
        x = self.conv(x)  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))  # apply prunable fully connected layer
        x = self.dropout(x)  # regularization
        return self.fc2(x)  