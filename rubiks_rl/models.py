import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        self.act = nn.ELU()
        self.fc1 = nn.Linear(54 * 6, 54 * 6)
        self.attn = nn.MultiheadAttention(embed_dim=54*6, num_heads=1, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=54*6)
    
    def forward(self, x):
        z = self.bn(x)
        z = self.fc1(x)
        z = self.act(z)
        z = self.bn(z)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.act(z)
        z = z + x
        return z


class RubikModel(nn.Module):
    def __init__(self, num_blocks: int):
        self.flatten = nn.Flatten()
        self.blocks = nn.Sequential(*[
            Block()
            for _ in range(num_blocks)
        ]) 

        self.value_head = nn.Linear(54 * 6, 1)
        self.policy_head = nn.Linear(54 * 6, 12)

    def forward(self, x):
        x = self.flatten(x)
        x = self.blocks(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p
