import torch
import torch.nn as nn
import rubiks_rl.constants as constants

from typing import Tuple


class Block(nn.Module):
    """
    General architecture taken from Karpathy's min GPT
    """

    def __init__(self, d_model: int, num_heads: int):
        super(Block, self).__init__()
        self.act = nn.GELU()
        self.fc_expand = nn.Linear(d_model, 4 * d_model)
        self.fc_compress = nn.Linear(4 * d_model, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        _, T, _ = x.size()
        z = self.ln1(x)
        z, _ = self.attn(z, z, z, need_weights=False, attn_mask=attn_mask)
        x = x + z
        x = x + self.fc_compress(self.act(self.fc_expand(x)))
        return x
    

class LMRubikModel(nn.Module):
    def __init__(self, num_blocks: int, d_model: int=192, num_heads: int=6):
        super(LMRubikModel, self).__init__()
        self.flatten = nn.Flatten()
        self.position_embed = nn.Embedding(constants.MAX_SEQUENCE_LENGTH, d_model)
        self.token_embed = nn.Embedding(20, d_model, padding_idx=constants.PADDING_IDX)  # (6 faces, 12 moves, 1 padding, 1 end of sequence)

        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads)
            for _ in range(num_blocks)
        ]) 

        self.next_move_head = nn.Sequential(
            nn.Linear(d_model, 12),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        device = x.device
        _, seq_length = x.size()
        pos = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embed(x)
        pos_embed = self.position_embed(pos)
        x = tok_emb + pos_embed
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.next_move_head(x)
        return x


class RLRubikModel(nn.Module):
    def __init__(self):
        super(RLRubikModel, self).__init__()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(54 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.value_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p
