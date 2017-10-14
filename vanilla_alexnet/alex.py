import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce

class AlexNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 12, 5)
    self.conv2 = nn.Conv2d(12, 16, 5)
    self.maxpool = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

    self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
    self.optim.zero_grad()

    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, x):
    x = self.maxpool(F.relu(self.conv1(x)))
    x = self.maxpool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def step(self, X, y):
    out = self(X)
    loss = self.loss_fn(out, y)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss

