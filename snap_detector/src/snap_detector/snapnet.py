import torch
import torch.nn as nn

class PerJointCNNGRUEncoder(nn.Module):
    def __init__(self, window_size, cnn_channels=16, rnn_hidden=32):
        super().__init__()
        self.cnn = nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1)
        self.rnn = nn.GRU(cnn_channels, rnn_hidden, batch_first=True)
        self.window_size = window_size

    def forward(self, x):  # x: (B, T, J)
        B, T, J = x.shape
        x = x.permute(0, 2, 1).reshape(B * J, 1, T)  # (B*J, 1, T)
        x = self.cnn(x)  # (B*J, C, T)
        x = x.permute(0, 2, 1)  # (B*J, T, C)
        _, h = self.rnn(x)  # (1, B*J, H)
        return h[-1].view(B, J, -1)  # (B, J, H)

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, J, H)
        scores = torch.softmax(self.attn(x), dim=1)  # (B, J, 1)
        return torch.sum(x * scores, dim=1)  # (B, H)

class SnapDetectorNet(nn.Module):
    def __init__(self, window_size, cnn_channels=32, rnn_hidden=32):
        super().__init__()
        self.encoder = PerJointCNNGRUEncoder(window_size, cnn_channels, rnn_hidden)
        self.pool = AttentionPooling(rnn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: (B, T, J)
        joint_features = self.encoder(x)
        pooled = self.pool(joint_features)
        return self.classifier(pooled).squeeze(1)