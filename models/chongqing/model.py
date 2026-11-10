import torch
import torch.nn as nn
import torch.nn.functional as F

class ChongqingDenseNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=71, stride=1, padding=35)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=71, stride=1, padding=35)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=71, stride=1, padding=35)
        self.conv4 = nn.Conv1d(96, 32, kernel_size=71, stride=1, padding=35)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv5 = nn.Conv1d(32, 24, kernel_size=71, stride=1, padding=35)
        self.conv6 = nn.Conv1d(56, 24, kernel_size=71, stride=1, padding=35)
        self.conv7 = nn.Conv1d(80, 24, kernel_size=71, stride=1, padding=35)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv8 = nn.Conv1d(24, 16, kernel_size=71, stride=1, padding=35)
        self.conv9 = nn.Conv1d(40, 16, kernel_size=71, stride=1, padding=35)
        self.conv10 = nn.Conv1d(56, 16, kernel_size=71, stride=1, padding=35)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (batch, 12, seq_len)
        x = self.pool1(self.relu(self.conv1(x)))
        
        d1 = self.relu(self.conv2(x))
        c1 = torch.cat([x, d1], dim=1)
        d2 = self.relu(self.conv3(c1))
        c2 = torch.cat([c1, d2], dim=1)
        d3 = self.relu(self.conv4(c2))
        x = self.pool2(d3)
        
        d4 = self.relu(self.conv5(x))
        c3 = torch.cat([x, d4], dim=1)
        d5 = self.relu(self.conv6(c3))
        c4 = torch.cat([c3, d5], dim=1)
        d6 = self.relu(self.conv7(c4))
        x = self.pool3(d6)
        
        d7 = self.relu(self.conv8(x))
        c5 = torch.cat([x, d7], dim=1)
        d8 = self.relu(self.conv9(c5))
        c6 = torch.cat([c5, d8], dim=1)
        d9 = self.relu(self.conv10(c6))
        
        out = self.gap(d9).squeeze(-1)
        logits = self.fc(out)
        return logits
