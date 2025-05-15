import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.video import r3d_18

class VAEFeatureProcessor(nn.Module):
    """
    VAEFeatureProcessor
    -------------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - in_channels (int): Number of input channels for the VAE features.
        - out_dim (int): Output feature dimension D after projection.
        - pretrained (bool): Whether to use a pretrained 3D ResNet backbone.
    Processes VAE features using a 3D ResNet backbone and projects to (B, T, D).
    Uses AdaptiveAvgPool3d((T,1,1)) to preserve the temporal dimension in a batch-wise, vectorized way.
    The projection layer is explicitly initialized with Xavier uniform.
    """
    def __init__(self, in_channels, out_dim, pretrained=False):
        super().__init__()
        self.resnet3d = r3d_18(pretrained=pretrained)
        if in_channels != 3:
            self.resnet3d.stem[0] = nn.Conv3d(
                in_channels, 64, kernel_size=(3, 7, 7),
                stride=(1, 2, 2), padding=(1, 3, 3), bias=False
            )
        self.resnet3d.avgpool = nn.Identity()
        self.resnet3d.fc = nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool only spatial dims, keep time
        self.proj = nn.Linear(512, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        feats = self.resnet3d(x)  # (B, 512, T, H', W')
        feats = self.pool(feats)  # (B, 512, T, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T, 512)
        out = self.proj(feats)  # (B, T, D)
        return out

class XDiTFeatureProcessor(nn.Module):
    """
    XDiTFeatureProcessor
    --------------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - in_channels (int): Number of input channels for the xDiT features.
        - out_dim (int): Output feature dimension D after projection.
    Processes xDiT features using Linear + BatchNorm, projects to (B, T, D).
    BatchNorm is applied independently for each time step in a vectorized way.
    The projection (linear) layer is explicitly initialized with Xavier uniform.
    """
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.mean(dim=[3, 4])
        x = x.permute(0, 2, 1)
        x = x.reshape(B * T, C)
        x = self.linear(x)
        x = self.bn(x)
        x = x.view(B, T, -1)
        return x

class FeatureFusion(nn.Module):
    """
    FeatureFusion
    -------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - in_dim_vae (int): Feature dimension of VAE input (D_vae).
        - in_dim_xdit (int): Feature dimension of xDiT input (D_xdit).
        - out_dim (int): Output fused feature dimension (D_fused).
        - dropout (float): Dropout probability after activation (default: 0.0).
        - activation (str): Activation function to use ('relu', 'gelu', etc.).
        - use_residual (bool): Whether to add a residual connection (default: False).
        - use_layernorm (bool): Whether to add LayerNorm after fusion (default: False).
    Concatenates VAE and xDiT features along the last dimension, then projects to out_dim,
    followed by BatchNorm, activation, Dropout, and optional residual/LayerNorm.
    If use_residual=True and input/output dims do not match, a linear mapping is used for the residual branch.
    Input: (B, T, D_vae), (B, T, D_xdit) -> Cat -> (B, T, D_vae + D_xdit) -> Proj -> (B, T, out_dim)
    """
    def __init__(self, in_dim_vae, in_dim_xdit, out_dim, dropout=0.0, activation='relu', use_residual=False, use_layernorm=False):
        super().__init__()
        self.in_dim_vae = in_dim_vae
        self.in_dim_xdit = in_dim_xdit
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.proj = nn.Linear(in_dim_vae + in_dim_xdit, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        if use_layernorm:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = None
        if use_residual and (in_dim_vae + in_dim_xdit) != out_dim:
            self.residual_proj = nn.Linear(in_dim_vae + in_dim_xdit, out_dim)
        else:
            self.residual_proj = None
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, vae_feat, xdit_feat):
        x = torch.cat([vae_feat, xdit_feat], dim=-1)  # (B, T, D_vae + D_xdit)
        B, T, D = x.shape
        proj_x = self.proj(x)                        # (B, T, out_dim)
        x_bn = self.bn(proj_x.view(B * T, -1)).view(B, T, -1)
        x_act = self.act(x_bn)
        x_drop = self.dropout(x_act)
        # Optional residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                res = self.residual_proj(x)
            else:
                res = x
            x_drop = x_drop + res
        # Optional LayerNorm (after Dropout)
        if self.ln is not None:
            x_drop = self.ln(x_drop)
        return x_drop

class TemporalEncoder(nn.Module):
    """
    TemporalEncoder
    ---------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - input_dim (int): Input feature dimension (should match D_fused).
        - encoder_type (str): 'gru' or 'transformer'.
        - nhead (int): Number of attention heads for transformer (default: 8).
        - num_layers (int): Number of layers for transformer (default: 2).
        - dropout (float): Dropout for transformer or GRU (default: 0.2).
        - max_len (int): Maximum sequence length for positional encoding (default: 512).
        - use_layernorm (bool): Whether to add LayerNorm after GRU output (default: False).
    Encodes temporal information using either a Bidirectional GRU or a Transformer Encoder.
    For GRU: output is projected back to input_dim, and optional LayerNorm is applied after output.
    For Transformer: LayerNorm is applied before and after the encoder, and a non-trainable positional encoding buffer is added (dynamically extended if needed).
    All switches should be passed as hyperparameters (e.g., via Hydra/CLI) for easier management and comparison.
    """
    def __init__(self, input_dim, encoder_type='gru', nhead=8, num_layers=2, dropout=0.2, max_len=512, use_layernorm=False):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.dropout = dropout
        self.max_len = max_len
        self.use_layernorm = use_layernorm
        if encoder_type == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=input_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )
            self.linear = nn.Linear(2 * input_dim, input_dim)
            if use_layernorm:
                self.ln = nn.LayerNorm(input_dim)
            else:
                self.ln = None
        elif encoder_type == 'transformer':
            self.ln_in = nn.LayerNorm(input_dim)
            self.ln_out = nn.LayerNorm(input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Register a non-trainable buffer for positional encoding, dynamically extended if needed
            self.register_buffer('pos_embed', torch.zeros(1, max_len, input_dim), persistent=False)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        if self.encoder_type == 'gru':
            out, _ = self.temporal_encoder(x)  # (B, T, 2*D)
            out = self.linear(out)             # (B, T, D)
            if self.use_layernorm and self.ln is not None:
                out = self.ln(out)
            return out
        elif self.encoder_type == 'transformer':
            x = self.ln_in(x)
            # Dynamically extend positional encoding buffer if needed
            if T > self.pos_embed.shape[1]:
                new_pos = torch.zeros(1, T, D, device=x.device, dtype=x.dtype)
                nn.init.trunc_normal_(new_pos, std=0.02)
                # Use buffer, not nn.Parameter
                self.register_buffer('pos_embed', new_pos, persistent=False)
            pos = self.pos_embed[:, :T, :]
            x = x + pos
            out = self.temporal_encoder(x)     # (B, T, D)
            out = self.ln_out(out)
            return out

        return out 