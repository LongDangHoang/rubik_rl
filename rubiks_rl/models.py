import torch
import torch.nn as nn

from typing import Tuple


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.act = nn.ELU()
        self.fc1 = nn.Linear(6, 6)
        self.attn = nn.MultiheadAttention(embed_dim=6, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm(6)
    
    def forward(self, x):
        z = self.fc1(x)
        z = self.act(z)
        z = self.ln(z)
        
        # cast N, E to N, 1, E and treat the sequence to have 
        z, _ = self.attn(z, z, z, need_weights=False)
        z = self.act(z)
        z = z + x
        return z


class RubikModel(nn.Module):
    def __init__(self, num_blocks: int):
        super(RubikModel, self).__init__()
        self.flatten = nn.Flatten()
        self.embed = nn.Linear(6, 6)
        self.blocks = nn.Sequential(*[
            Block()
            for _ in range(num_blocks)
        ]) 

        self.value_head = nn.Linear(54 * 6, 1)
        self.policy_head = nn.Sequential(
            nn.Linear(54 * 6, 12),
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        x = self.blocks(x)
        x = self.flatten(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p
