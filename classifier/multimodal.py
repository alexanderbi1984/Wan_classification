import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.video import r3d_18
import torchmetrics
from torchmetrics.classification import CohenKappa, Accuracy

class VAEFeatureProcessor(nn.Module):
    """
    VAEFeatureProcessor
    -------------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - in_channels (int): Number of input channels for the VAE features.
        - out_dim (int): Output feature dimension D after projection.
        - pretrained (bool): Whether to use a pretrained 3D ResNet backbone.
    Uses a 3D ResNet-18 backbone with all temporal strides forced to 1,
    so the time dimension T remains unchanged. Applies spatial pooling
    and a linear projection to produce (B, T, D) features.
    """
    def __init__(self, in_channels: int, out_dim: int, pretrained: bool = False):
        super().__init__()
        # Load pretrained 3D ResNet-18
        base = r3d_18(pretrained=pretrained)
        # Adjust input channels if necessary
        if in_channels != 3:
            base.stem[0] = nn.Conv3d(
                in_channels, 64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            )
        # Remove final pooling & fc layers
        modules = list(base.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        # Force all Conv3d temporal strides to 1 to preserve T
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv3d):
                s = m.stride
                m.stride = (1, s[1], s[2])
        # Spatial pooling only (keep time dim)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Final projection from 512 to out_dim
        self.proj = nn.Linear(512, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        feats = self.backbone(x)  # (B, 512, T, H', W')
        # Temporarily disable deterministic checks for pooling
        prev = torch.are_deterministic_algorithms_enabled()
        # torch.use_deterministic_algorithms(False)
        feats = self.pool(feats)  # (B, 512, T, 1, 1)
        # torch.use_deterministic_algorithms(prev)
        # Collapse spatial dims, keep time
        feats = feats.squeeze(-1).squeeze(-1)     # (B, 512, T)
        feats = feats.transpose(1, 2)             # (B, T, 512)
        # Project to (B, T, D)
        out = self.proj(feats)
        return out



    # def __init__(self, in_channels, out_dim, pretrained=False):
    #     super().__init__()
    #     self.resnet3d = r3d_18(pretrained=pretrained)
    #     if in_channels != 3:
    #         self.resnet3d.stem[0] = nn.Conv3d(
    #             in_channels, 64, kernel_size=(3, 7, 7),
    #             stride=(1, 2, 2), padding=(1, 3, 3), bias=False
    #         )
    #     self.resnet3d.avgpool = nn.Identity()
    #     self.resnet3d.fc = nn.Identity()
    #     self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool only spatial dims, keep time
    #     self.proj = nn.Linear(512, out_dim)
    #     nn.init.xavier_uniform_(self.proj.weight)
    #     nn.init.zeros_(self.proj.bias)

    # def forward(self, x):
    #     print("[DEBUG] VAEFeatureProcessor input type:", type(x), "shape:", getattr(x, 'shape', None))
    #     assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)}"
    #     assert x.ndim >= 4, f"VAEFeatureProcessor input shape: {x.shape}"
    #     feats = self.resnet3d(x)  # (B, 512, T, H', W') or (B, 512, T) or (B, 512, T, 1, 1)
    #     print("[DEBUG] feats shape after resnet3d:", feats.shape)
    #     # Ensure feats is 5D for AdaptiveAvgPool3d
    #     if feats.dim() == 3:
    #         # (B, 512, T) -> (B, 512, T, 1, 1)
    #         feats = feats.unsqueeze(-1).unsqueeze(-1)
    #     elif feats.dim() == 4:
    #         # (B, 512, T, 1) -> (B, 512, T, 1, 1)
    #         feats = feats.unsqueeze(-1)
    #     # Only pool if not already pooled spatially
    #     if feats.shape[-2:] != (1, 1):
    #         feats = self.pool(feats)  # (B, 512, T, 1, 1)
    #     feats = feats.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T, 512)
    #     out = self.proj(feats)  # (B, T, D)
    #     return out

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

class TemporalPooling(nn.Module):
    """
    TemporalPooling
    ---------------
    Configurable hyperparameters (recommended to manage via Hydra/CLI):
        - pooling_type (str): 'mean', 'max', or 'cls'.
    Pools the temporal dimension of a sequence tensor (B, T, D) to (B, D).
    'mean': Mean pooling over T.
    'max': Max pooling over T.
    'cls': Use the first time step (e.g., for transformer with [CLS] token).
    """
    def __init__(self, pooling_type='mean'):
        super().__init__()
        assert pooling_type in ['mean', 'max', 'cls'], f"Unsupported pooling_type: {pooling_type}"
        self.pooling_type = pooling_type

    def forward(self, x):
        # x: (B, T, D)
        if self.pooling_type == 'mean':
            return x.mean(dim=1)
        elif self.pooling_type == 'max':
            return x.max(dim=1)[0]
        elif self.pooling_type == 'cls':
            return x[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

class MultimodalMultiTaskCoralClassifier(pl.LightningModule):
    """
    Multimodal Multi-Task CORAL Classifier
    -------------------------------------
    Processes VAE and xDiT features, fuses them, encodes temporally, pools, applies a shared MLP encoder, and predicts multitask ordinal outputs.
    Configurable via Hydra/CLI for all major hyperparameters.
    """
    def __init__(self,
                 vae_in_channels,
                 xdit_in_channels,
                 feature_dim,
                 fusion_dim,
                 temporal_encoder_dim,  # Not used, kept for config compatibility
                 num_pain_classes,
                 num_stimulus_classes,
                 shared_mlp_hidden_dims=None,
                 shared_mlp_dropout=0.5,
                 fusion_dropout=0.0,
                 fusion_activation='relu',
                 fusion_use_residual=False,
                 fusion_use_layernorm=False,
                 temporal_encoder_type='gru',
                 temporal_encoder_nhead=8,
                 temporal_encoder_num_layers=2,
                 temporal_encoder_dropout=0.2,
                 temporal_encoder_max_len=512,
                 temporal_encoder_use_layernorm=False,
                 temporal_pooling_type='mean',
                 learning_rate=1e-4,
                 optimizer_name='AdamW',
                 pain_loss_weight=1.0,
                 stim_loss_weight=1.0,
                 weight_decay=0.0,
                 label_smoothing=0.0,
                 use_distance_penalty=False,
                 focal_gamma=None,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        # Feature processors
        self.vae_processor = VAEFeatureProcessor(vae_in_channels, feature_dim)
        self.xdit_processor = XDiTFeatureProcessor(xdit_in_channels, feature_dim)

        # Fusion
        self.fusion = FeatureFusion(
            in_dim_vae=feature_dim,
            in_dim_xdit=feature_dim,
            out_dim=fusion_dim,
            dropout=fusion_dropout,
            activation=fusion_activation,
            use_residual=fusion_use_residual,
            use_layernorm=fusion_use_layernorm
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=fusion_dim,
            encoder_type=temporal_encoder_type,
            nhead=temporal_encoder_nhead,
            num_layers=temporal_encoder_num_layers,
            dropout=temporal_encoder_dropout,
            max_len=temporal_encoder_max_len,
            use_layernorm=temporal_encoder_use_layernorm
        )

        # Temporal pooling
        self.temporal_pooling = TemporalPooling(pooling_type=temporal_pooling_type)

        # Shared MLP encoder (mandatory)
        if shared_mlp_hidden_dims is None or len(shared_mlp_hidden_dims) == 0:
            self.shared_encoder = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(shared_mlp_dropout)
            )
            encoder_output_dim = fusion_dim
        else:
            layers = []
            current_dim = fusion_dim
            for h_dim in shared_mlp_hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(shared_mlp_dropout))
                current_dim = h_dim
            self.shared_encoder = nn.Sequential(*layers)
            encoder_output_dim = current_dim

        # CORAL heads
        self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
        self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)

        # Store class weights for weighted loss calculation
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Metrics (MAE, QWK, Accuracy) for both tasks
        metric_args = {'dist_sync_on_step': False}
        self.train_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.train_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.train_pain_qwk = CohenKappa(task="multiclass", num_classes=num_pain_classes, weights='quadratic', **metric_args)
        self.val_pain_qwk = CohenKappa(task="multiclass", num_classes=num_pain_classes, weights='quadratic', **metric_args)
        self.test_pain_qwk = CohenKappa(task="multiclass", num_classes=num_pain_classes, weights='quadratic', **metric_args)
        self.train_stim_qwk = CohenKappa(task="multiclass", num_classes=num_stimulus_classes, weights='quadratic', **metric_args)
        self.val_stim_qwk = CohenKappa(task="multiclass", num_classes=num_stimulus_classes, weights='quadratic', **metric_args)
        self.test_stim_qwk = CohenKappa(task="multiclass", num_classes=num_stimulus_classes, weights='quadratic', **metric_args)
        self.train_pain_acc = Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        self.val_pain_acc = Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        self.test_pain_acc = Accuracy(task="multiclass", num_classes=num_pain_classes, average='macro', **metric_args)
        self.train_stim_acc = Accuracy(task="multiclass", num_classes=num_stimulus_classes, average='macro', **metric_args)
        self.val_stim_acc = Accuracy(task="multiclass", num_classes=num_stimulus_classes, average='macro', **metric_args)
        self.test_stim_acc = Accuracy(task="multiclass", num_classes=num_stimulus_classes, average='macro', **metric_args)

    @staticmethod
    def coral_loss(logits, levels, importance_weights=None, reduction='mean', label_smoothing=0.0, distance_penalty=False, focal_gamma=None):
        if logits.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        num_classes_minus1 = logits.shape[1]
        levels = levels.long()
        levels_binary = (levels.unsqueeze(1) > torch.arange(num_classes_minus1, device=logits.device).unsqueeze(0)).float()
        if label_smoothing > 0.0:
            levels_binary = levels_binary * (1 - label_smoothing) + label_smoothing / 2
        log_sigmoid = torch.nn.functional.logsigmoid(logits)
        base_loss_tasks = log_sigmoid * levels_binary + (log_sigmoid - logits) * (1 - levels_binary)
        if distance_penalty:
            distance_matrix = torch.abs(
                levels.unsqueeze(1) - torch.arange(num_classes_minus1, device=logits.device).unsqueeze(0)
            ).float()
            base_loss_tasks = base_loss_tasks * (1.0 + distance_matrix)
        if focal_gamma is not None:
            probs = torch.sigmoid(logits)
            eps = 1e-6
            focal_weight = torch.where(
                levels_binary > 0.5,
                (1 - probs + eps) ** focal_gamma,
                (probs + eps) ** focal_gamma
            )
            base_loss_tasks = focal_weight * base_loss_tasks
        loss_per_sample = -torch.sum(base_loss_tasks, dim=1)
        if importance_weights is not None:
            loss_per_sample *= importance_weights
        if reduction == 'mean':
            return loss_per_sample.mean()
        elif reduction == 'sum':
            return loss_per_sample.sum()
        elif reduction == 'none':
            return loss_per_sample
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

    @staticmethod
    def prob_to_label(probs, num_classes=None):
        labels = torch.sum(probs > 0.5, dim=1)
        if num_classes is not None:
            labels = torch.clamp(labels, max=num_classes - 1)
        return labels

    def forward(self, vae_x, xdit_x):
        # vae_x, xdit_x: (B, C, T, H, W)
        vae_feat = self.vae_processor(vae_x)      # (B, T, D)
        xdit_feat = self.xdit_processor(xdit_x)   # (B, T, D)
        fused = self.fusion(vae_feat, xdit_feat)  # (B, T, fusion_dim)
        encoded = self.temporal_encoder(fused)    # (B, T, fusion_dim)
        pooled = self.temporal_pooling(encoded)   # (B, fusion_dim)
        shared = self.shared_encoder(pooled)      # (B, encoder_output_dim)
        pain_logits = self.pain_head(shared)      # (B, num_pain_classes-1)
        stim_logits = self.stimulus_head(shared)  # (B, num_stimulus_classes-1)
        return pain_logits, stim_logits

    def _update_metrics(self, pain_logits, stim_logits, pain_labels, stim_labels, stage):
        # Pain task
        valid_pain_mask = pain_labels != -1
        if valid_pain_mask.any():
            valid_pain_logits = pain_logits[valid_pain_mask]
            valid_pain_labels = pain_labels[valid_pain_mask]
            pain_probs = torch.sigmoid(valid_pain_logits)
            pain_preds = self.prob_to_label(pain_probs, num_classes=self.hparams.num_pain_classes)
            getattr(self, f"{stage}_pain_mae").update(pain_preds, valid_pain_labels)
            getattr(self, f"{stage}_pain_qwk").update(pain_preds, valid_pain_labels)
            getattr(self, f"{stage}_pain_acc").update(pain_preds, valid_pain_labels)
        # Stimulus task
        valid_stim_mask = stim_labels != -1
        if valid_stim_mask.any():
            valid_stim_logits = stim_logits[valid_stim_mask]
            valid_stim_labels = stim_labels[valid_stim_mask]
            stim_probs = torch.sigmoid(valid_stim_logits)
            stim_preds = self.prob_to_label(stim_probs, num_classes=self.hparams.num_stimulus_classes)
            getattr(self, f"{stage}_stim_mae").update(stim_preds, valid_stim_labels)
            getattr(self, f"{stage}_stim_qwk").update(stim_preds, valid_stim_labels)
            getattr(self, f"{stage}_stim_acc").update(stim_preds, valid_stim_labels)

    def training_step(self, batch, batch_idx):
        vae_x, xdit_x, pain_labels, stim_labels = batch
        pain_logits, stim_logits = self(vae_x, xdit_x)
        # Pain loss
        valid_pain_mask = pain_labels != -1
        if valid_pain_mask.any():
            valid_pain_logits = pain_logits[valid_pain_mask]
            valid_pain_labels = pain_labels[valid_pain_mask]
            importance_weights = self.class_weights[valid_pain_labels] if self.class_weights is not None else None
            pain_loss = self.coral_loss(
                valid_pain_logits, valid_pain_labels,
                importance_weights=importance_weights,
                label_smoothing=self.hparams.label_smoothing,
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma
            )
        else:
            pain_loss = torch.tensor(0.0, device=self.device)
        # Stimulus loss
        valid_stim_mask = stim_labels != -1
        if valid_stim_mask.any():
            valid_stim_logits = stim_logits[valid_stim_mask]
            valid_stim_labels = stim_labels[valid_stim_mask]
            stim_loss = self.coral_loss(
                valid_stim_logits, valid_stim_labels,
                label_smoothing=0.0,
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma
            )
        else:
            stim_loss = torch.tensor(0.0, device=self.device)
        total_loss = self.hparams.pain_loss_weight * pain_loss + self.hparams.stim_loss_weight * stim_loss
        self._update_metrics(pain_logits, stim_logits, pain_labels, stim_labels, stage='train')
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pain_MAE', self.train_pain_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pain_QWK', self.train_pain_qwk, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pain_Accuracy', self.train_pain_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_stim_MAE', self.train_stim_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_stim_QWK', self.train_stim_qwk, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_stim_Accuracy', self.train_stim_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        vae_x, xdit_x, pain_labels, stim_labels = batch
        pain_logits, stim_logits = self(vae_x, xdit_x)
        self._update_metrics(pain_logits, stim_logits, pain_labels, stim_labels, stage='val')
        # Log validation metrics for early stopping and progress bar
        self.log('val_pain_MAE', self.val_pain_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pain_QWK', self.val_pain_qwk, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pain_Accuracy', self.val_pain_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_stim_MAE', self.val_stim_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_stim_QWK', self.val_stim_qwk, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_stim_Accuracy', self.val_stim_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {}

    def test_step(self, batch, batch_idx):
        vae_x, xdit_x, pain_labels, stim_labels = batch
        pain_logits, stim_logits = self(vae_x, xdit_x)
        self._update_metrics(pain_logits, stim_logits, pain_labels, stim_labels, stage='test')
        return {}

    def configure_optimizers(self):
        lr = float(self.hparams.learning_rate)
        wd = float(self.hparams.weight_decay)
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        # LR scheduler support (ReduceLROnPlateau)
        if hasattr(self.hparams, 'use_lr_scheduler') and self.hparams.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if hasattr(self.hparams, 'monitor_metric') and 'QWK' in self.hparams.monitor_metric else 'min',
                factor=getattr(self.hparams, 'lr_factor', 0.5),
                patience=getattr(self.hparams, 'lr_patience', 5),
                min_lr=getattr(self.hparams, 'min_lr', 1e-6)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": getattr(self.hparams, 'monitor_metric', 'val_pain_QWK'),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                }
            }
        return optimizer

    def sanity_check(self, batch_size: int = 4, C=3, T=8, H=16, W=16):
        """Performs a basic check on input/output shapes."""
        print(f"Running sanity check for {self.__class__.__name__}...")
        try:
            device = self.device
            dummy_vae = torch.randn(batch_size, C, T, H, W, device=device)
            dummy_xdit = torch.randn(batch_size, C, T, H, W, device=device)
            dummy_pain = torch.randint(0, self.hparams.num_pain_classes, (batch_size,), device=device)
            dummy_stim = torch.randint(0, self.hparams.num_stimulus_classes, (batch_size,), device=device)
            self.eval()
            with torch.no_grad():
                pain_logits, stim_logits = self(dummy_vae, dummy_xdit)
            self.train()
            expected_pain_shape = (batch_size, self.hparams.num_pain_classes - 1)
            expected_stim_shape = (batch_size, self.hparams.num_stimulus_classes - 1)
            assert pain_logits.shape == expected_pain_shape, \
                f"Pain logits shape mismatch: Expected {expected_pain_shape}, Got {pain_logits.shape}"
            assert stim_logits.shape == expected_stim_shape, \
                f"Stimulus logits shape mismatch: Expected {expected_stim_shape}, Got {stim_logits.shape}"
            print(f"Sanity check passed for {self.__class__.__name__}.")
            print(f"  VAE input shape: {dummy_vae.shape}")
            print(f"  xDiT input shape: {dummy_xdit.shape}")
            print(f"  Pain logits shape: {pain_logits.shape}")
            print(f"  Stimulus logits shape: {stim_logits.shape}")
            print(f"  Pain head weights: {self.pain_head.weight.data.norm():.4f}")
            print(f"  Stim head weights: {self.stimulus_head.weight.data.norm():.4f}")
        except Exception as e:
            print(f"Sanity check failed for {self.__class__.__name__}: {e}")
            raise

class BioVidMultimodalCoralClassifier(MultimodalMultiTaskCoralClassifier):
    """
    A version of MultimodalMultiTaskCoralClassifier specifically for the BioVid dataset.
    Ensures LR scheduler uses stimulus metrics (val_stim_QWK) instead of pain metrics.
    """
    def configure_optimizers(self):
        try:
            lr = float(self.hparams.learning_rate)
            wd = float(self.hparams.weight_decay)
        except ValueError:
            print(f"Error: Could not convert learning rate '{self.hparams.learning_rate}' to float!")
            raise
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        if hasattr(self.hparams, 'use_lr_scheduler') and self.hparams.use_lr_scheduler:
            monitor_metric = "val_stim_QWK"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # QWK is always maximized
                factor=getattr(self.hparams, 'lr_factor', 0.5),
                patience=getattr(self.hparams, 'lr_patience', 5),
                min_lr=getattr(self.hparams, 'min_lr', 1e-6)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                }
            }
        return optimizer 