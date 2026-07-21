import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectSchedulerNet(nn.Module):
    def __init__(self, feature_dim: int = 3, embed_dim: int = 64, num_heads: int = 4, queue_size: int = 10):
        super(DirectSchedulerNet, self).__init__()
        self.embed_dim = embed_dim
        self.queue_size = queue_size

        # Input Embedding Layer
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, queue_size, embed_dim) * 0.01)

        # Transformer Self-Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Policy Output Head: Produces single Q-value scalar per process slot
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, N, feature_dim)
        mask: (Batch, N) - True for valid process, False for padded slot
        """
        batch_size, N, _ = x.shape
        
        # Project state to embedding space
        h = self.input_proj(x) + self.pos_embedding[:, :N, :]
        
        # Key Padding Mask inside Attention Module (PyTorch expects True for padded/ignored tokens)
        key_padding_mask = ~mask  # Invert: True where padded
        
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.layer_norm(h + attn_out)

        # Q-value evaluation per slot: (Batch, N, 1) -> (Batch, N)
        q_values = self.q_head(h).squeeze(-1)

        # Force Q-values of padded slots to -inf to ensure validity during action selection
        q_values = q_values.masked_fill(~mask, float('-inf'))
        return q_values