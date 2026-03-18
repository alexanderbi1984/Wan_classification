"""
Online Multimodal Multi-Task Classifier with LoRA fine-tuning on the Wan DiT backbone.

Replaces the offline pre-extracted feature pipeline with end-to-end training:
    Raw video -> WanVAE (frozen) -> WanModel/DiT (LoRA) -> downstream heads

Reuses existing components:
    - VAEFeatureProcessor, XDiTFeatureProcessor, FeatureFusion
    - TemporalEncoder, TemporalPooling
    - CORAL ordinal classification heads
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import CohenKappa, Accuracy, MulticlassF1Score, MulticlassConfusionMatrix

from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.vae import WanVAE
from feature_extraction.extract_xdit_features import FeatureExtractorWanModel
from classifier.multimodal import (
    VAEFeatureProcessor,
    XDiTFeatureProcessor,
    FeatureFusion,
    TemporalEncoder,
    TemporalPooling,
    MultimodalMultiTaskCoralClassifier,
)
from utils.lora import (
    inject_lora_into_wan_model,
    prepare_lora_training,
    enable_gradient_checkpointing,
    save_lora_weights,
)


class OnlineMultimodalClassifier(pl.LightningModule):
    """End-to-end classifier that runs Wan encoder online with LoRA.

    Architecture:
        video (B, C, T, H, W)
            -> WanVAE.encode() [frozen, no_grad] -> vae_latents (B, 16, T', H', W')
            -> WanModel/DiT [LoRA on attention] -> dit_features (B, dim, T', H', W')
            -> VAEFeatureProcessor + XDiTFeatureProcessor
            -> FeatureFusion -> TemporalEncoder -> TemporalPooling
            -> Shared MLP -> CORAL heads (pain, stimulus)

    Args:
        wan_checkpoint_dir: Path to WanModel checkpoint directory.
        vae_checkpoint: Path to Wan2.1_VAE.pth file.
        lora_rank: LoRA adapter rank.
        lora_alpha: LoRA scaling factor.
        lora_target_modules: Which attention sub-modules to apply LoRA to.
        use_gradient_checkpointing: Enable activation checkpointing for memory savings.
        vae_in_channels: VAE latent channels (default 16).
        xdit_in_channels: DiT feature channels (model dim, e.g. 1536 for 1.3B).
        feature_dim: Intermediate feature dimension after processors.
        fusion_dim: Fused feature dimension.
        num_pain_classes: Number of ordinal pain classes.
        num_stimulus_classes: Number of ordinal stimulus classes.
        shared_mlp_hidden_dims: Hidden dimensions for the shared MLP.
        shared_mlp_dropout: Dropout in the shared MLP.
        fusion_dropout: Dropout in FeatureFusion.
        fusion_activation: Activation in FeatureFusion ('relu' or 'gelu').
        fusion_use_residual: Whether to use residual in FeatureFusion.
        fusion_use_layernorm: Whether to use LayerNorm in FeatureFusion.
        temporal_encoder_type: 'gru' or 'transformer'.
        temporal_encoder_nhead: Number of heads for transformer temporal encoder.
        temporal_encoder_num_layers: Number of layers for temporal encoder.
        temporal_encoder_dropout: Dropout for temporal encoder.
        temporal_encoder_max_len: Maximum sequence length for positional encoding.
        temporal_encoder_use_layernorm: LayerNorm flag for temporal encoder.
        temporal_pooling_type: 'mean', 'max', or 'cls'.
        lr_backbone: Learning rate for LoRA parameters.
        lr_head: Learning rate for downstream head parameters.
        weight_decay: Weight decay for AdamW.
        pain_loss_weight: Loss weight for pain task.
        stim_loss_weight: Loss weight for stimulus task.
        label_smoothing: CORAL label smoothing.
        use_distance_penalty: Distance penalty in CORAL loss.
        focal_gamma: Focal loss gamma (None to disable).
        dit_feature_layer: Which DiT block to extract features from (-1 = last).
        dit_feature_layers: List of DiT block indices for multi-layer extraction.
            When provided, features from these blocks are combined via learnable
            scalar weights (like BERT's scalar mix). Overrides dit_feature_layer.
        prompt_embeddings_path: Path to pre-computed T5 prompt embeddings (.pt).
            When provided, the DiT cross-attention receives real text context
            (randomly sampled per batch during training, fixed during eval)
            instead of a dummy zero tensor.
        coral_alpha: Weight for CORAL ordinal loss (default 1.0 for backward compat).
        ce_alpha: Weight for CE classification loss (default 0.0 = CORAL-only).
            When > 0, a parallel CE head (K outputs) is added alongside CORAL.
        eval_head: Which head to use for metrics: 'coral' or 'ce'.
        ce_focal_gamma: Focal loss gamma for CE head (None to disable).
        ce_label_smoothing: Label smoothing for CE loss (default 0.0).
        mixup_alpha: Beta distribution parameter for MixUp (default 0.0 = disabled).
            When > 0, blends pairs of training videos at the pixel level to
            disrupt identity cues and improve generalization.
    """

    def __init__(
        self,
        wan_checkpoint_dir: str,
        vae_checkpoint: str,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_target_modules: list = None,
        use_gradient_checkpointing: bool = False,
        vae_in_channels: int = 16,
        xdit_in_channels: int = 1536,
        feature_dim: int = 128,
        fusion_dim: int = 128,
        num_pain_classes: int = 5,
        num_stimulus_classes: int = 4,
        shared_mlp_hidden_dims: list = None,
        shared_mlp_dropout: float = 0.5,
        fusion_dropout: float = 0.0,
        fusion_activation: str = "relu",
        fusion_use_residual: bool = False,
        fusion_use_layernorm: bool = False,
        temporal_encoder_type: str = "gru",
        temporal_encoder_nhead: int = 8,
        temporal_encoder_num_layers: int = 2,
        temporal_encoder_dropout: float = 0.2,
        temporal_encoder_max_len: int = 512,
        temporal_encoder_use_layernorm: bool = False,
        temporal_pooling_type: str = "mean",
        spatial_pool: str = "mean",
        lr_backbone: float = 5e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        pain_loss_weight: float = 1.0,
        stim_loss_weight: float = 1.0,
        label_smoothing: float = 0.0,
        use_distance_penalty: bool = False,
        focal_gamma: float = None,
        dit_feature_layer: int = -1,
        dit_feature_layers: list = None,
        dit_timestep: float = 0.0,
        prompt_embeddings_path: str = None,
        coral_alpha: float = 1.0,
        ce_alpha: float = 0.0,
        eval_head: str = "coral",
        ce_focal_gamma: float = None,
        ce_label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ----- Upstream: VAE (frozen) -----
        # WanVAE is NOT an nn.Module — it's a plain wrapper.  We extract
        # its inner nn.Module (.model) so Lightning can track device moves,
        # and register the normalization scale as buffers for device sync.
        _vae_wrapper = WanVAE(vae_pth=vae_checkpoint)
        self.vae_model = _vae_wrapper.model       # nn.Module, already eval + frozen
        self.register_buffer("_vae_scale_mean", _vae_wrapper.scale[0], persistent=False)
        self.register_buffer("_vae_scale_inv_std", _vae_wrapper.scale[1], persistent=False)
        self._vae_dtype = _vae_wrapper.dtype

        # ----- Upstream: DiT backbone (with LoRA) -----
        dit_model = WanModel.from_pretrained(wan_checkpoint_dir)
        self.dit_dim = dit_model.dim

        if use_gradient_checkpointing:
            enable_gradient_checkpointing(dit_model)

        lora_params = inject_lora_into_wan_model(
            dit_model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        trainable_lora = prepare_lora_training(dit_model)
        print(f"[OnlineMultimodalClassifier] LoRA injected: {lora_params:,} param elements in adapters, "
              f"{trainable_lora:,} trainable")

        self.feature_extractor = FeatureExtractorWanModel(dit_model)

        # ----- Multi-layer feature extraction (optional) -----
        self._multi_layer = dit_feature_layers is not None and len(dit_feature_layers) > 1
        if self._multi_layer:
            n_layers = len(dit_feature_layers)
            self.layer_weights = nn.Parameter(torch.zeros(n_layers))
            self.layer_gamma = nn.Parameter(torch.ones(1))
            print(f"[OnlineMultimodalClassifier] Multi-layer DiT extraction from "
                  f"blocks {dit_feature_layers} ({n_layers} layers, learnable scalar mix)")

        # ----- Pre-computed prompt embeddings (optional) -----
        self._prompt_embeddings = None
        self._prompt_seq_lens = None
        if prompt_embeddings_path is not None:
            data = torch.load(prompt_embeddings_path, map_location="cpu")
            embeds = data["embeddings"]   # list of (seq_len_i, 4096) float32
            self._prompt_seq_lens = data["seq_lens"]
            # Register as non-persistent buffers so they move with the model
            for i, emb in enumerate(embeds):
                self.register_buffer(f"_prompt_emb_{i}", emb, persistent=False)
            self._prompt_embeddings = len(embeds)
            print(f"[OnlineMultimodalClassifier] Loaded {self._prompt_embeddings} "
                  f"prompt embeddings from {prompt_embeddings_path}")

        # Override xdit_in_channels to match the actual DiT dim
        xdit_in_channels = self.dit_dim

        # ----- Downstream head (reuses existing components) -----
        self.vae_processor = VAEFeatureProcessor(vae_in_channels, feature_dim)
        spatial_pool = self.hparams.get("spatial_pool", "mean")
        self.xdit_processor = XDiTFeatureProcessor(
            xdit_in_channels, feature_dim, spatial_pool=spatial_pool
        )

        self.fusion = FeatureFusion(
            in_dim_vae=feature_dim,
            in_dim_xdit=feature_dim,
            out_dim=fusion_dim,
            dropout=fusion_dropout,
            activation=fusion_activation,
            use_residual=fusion_use_residual,
            use_layernorm=fusion_use_layernorm,
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=fusion_dim,
            encoder_type=temporal_encoder_type,
            nhead=temporal_encoder_nhead,
            num_layers=temporal_encoder_num_layers,
            dropout=temporal_encoder_dropout,
            max_len=temporal_encoder_max_len,
            use_layernorm=temporal_encoder_use_layernorm,
        )

        self.temporal_pooling = TemporalPooling(
            pooling_type=temporal_pooling_type,
            input_dim=fusion_dim,
        )

        # Shared MLP encoder
        if shared_mlp_hidden_dims is None or len(shared_mlp_hidden_dims) == 0:
            self.shared_encoder = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(shared_mlp_dropout),
            )
            encoder_output_dim = fusion_dim
        else:
            layers = []
            current_dim = fusion_dim
            for h_dim in shared_mlp_hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                    nn.Dropout(shared_mlp_dropout),
                ])
                current_dim = h_dim
            self.shared_encoder = nn.Sequential(*layers)
            encoder_output_dim = current_dim

        # CORAL ordinal heads (K-1 threshold logits)
        self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
        self._use_stim = (stim_loss_weight > 0.0)
        if self._use_stim:
            self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)
        else:
            self.stimulus_head = None

        # CE classification heads (K class logits) — only created when ce_alpha > 0
        self._use_ce = ce_alpha > 0.0
        if self._use_ce:
            self.ce_pain_head = nn.Linear(encoder_output_dim, num_pain_classes)
            if self._use_stim:
                self.ce_stim_head = nn.Linear(encoder_output_dim, num_stimulus_classes)
            print(f"[OnlineMultimodalClassifier] Hybrid loss: "
                  f"coral_alpha={coral_alpha}, ce_alpha={ce_alpha}, eval_head={eval_head}")

        # ----- Metrics -----
        metric_args = {"dist_sync_on_step": False}
        for stage in ("train", "val", "test"):
            for task, n_cls in [("pain", num_pain_classes), ("stim", num_stimulus_classes)]:
                setattr(self, f"{stage}_{task}_mae", torchmetrics.MeanAbsoluteError(**metric_args))
                setattr(self, f"{stage}_{task}_qwk", CohenKappa(task="multiclass", num_classes=n_cls, weights="quadratic", **metric_args))
                setattr(self, f"{stage}_{task}_acc", Accuracy(task="multiclass", num_classes=n_cls, average="macro", **metric_args))
                setattr(self, f"{stage}_{task}_f1", MulticlassF1Score(num_classes=n_cls, average="macro", **metric_args))
                # Per-class recall and confusion matrix for val/test
                if stage in ("val", "test"):
                    setattr(self, f"{stage}_{task}_recall_per_class",
                            torchmetrics.classification.MulticlassRecall(
                                num_classes=n_cls, average="none", **metric_args))
                    setattr(self, f"{stage}_{task}_cm",
                            MulticlassConfusionMatrix(
                                num_classes=n_cls, **metric_args))

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def forward(self, video_tensor: torch.Tensor):
        """End-to-end forward: raw video -> logits.

        Args:
            video_tensor: (B, C, T, H, W) float32 video normalized to [0, 1].

        Returns:
            dict with keys:
                'pain_coral': (B, K_pain - 1) CORAL logits
                'stim_coral': (B, K_stim - 1) CORAL logits
                'pain_ce':    (B, K_pain) CE logits  (only when ce_alpha > 0)
                'stim_ce':    (B, K_stim) CE logits  (only when ce_alpha > 0)
        """
        # VAE encoding (frozen, no gradient)
        vae_scale = [self._vae_scale_mean, self._vae_scale_inv_std]
        with torch.no_grad(), amp.autocast(dtype=self._vae_dtype):
            vae_latents = torch.stack([
                self.vae_model.encode(
                    video_tensor[i].unsqueeze(0), vae_scale
                ).float().squeeze(0)
                for i in range(video_tensor.shape[0])
            ])  # (B, 16, T', H', W')

        # DiT feature extraction (LoRA has gradients)
        dit_features = self._extract_dit_features(vae_latents)  # (B, dim, T', H', W')

        # Downstream pipeline
        vae_feat = self.vae_processor(vae_latents)       # (B, T_vae, D)
        xdit_feat = self.xdit_processor(dit_features)    # (B, T_xdit, D)

        # Align temporal dims (r3d_18 downsamples T; DiT does not)
        if vae_feat.shape[1] != xdit_feat.shape[1]:
            vae_feat = F.interpolate(
                vae_feat.transpose(1, 2),          # (B, D, T_vae)
                size=xdit_feat.shape[1],
                mode='nearest'
            ).transpose(1, 2)                      # (B, T_xdit, D)

        fused = self.fusion(vae_feat, xdit_feat)         # (B, T, fusion_dim)
        encoded = self.temporal_encoder(fused)            # (B, T, fusion_dim)
        pooled = self.temporal_pooling(encoded)           # (B, fusion_dim)
        shared = self.shared_encoder(pooled)              # (B, encoder_output_dim)

        out = {"pain_coral": self.pain_head(shared)}
        if self._use_stim:
            out["stim_coral"] = self.stimulus_head(shared)
        if self._use_ce:
            out["pain_ce"] = self.ce_pain_head(shared)
            if self._use_stim:
                out["stim_ce"] = self.ce_stim_head(shared)
        return out

    def _extract_dit_features(self, vae_latents: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from DiT with LoRA.

        Args:
            vae_latents: (B, 16, T', H', W') VAE-encoded latents.

        Returns:
            (B, dim, T', H', W') spatial-temporal features from the DiT.
        """
        model = self.feature_extractor.model
        B = vae_latents.shape[0]
        device = vae_latents.device

        t = torch.full((B,), self.hparams.dit_timestep, device=device)

        # Build text context: real prompt embeddings or dummy zeros
        if self._prompt_embeddings is not None:
            if self.training:
                idx = torch.randint(0, self._prompt_embeddings, (1,)).item()
            else:
                idx = 0
            prompt_emb = getattr(self, f"_prompt_emb_{idx}").to(
                device=device, dtype=vae_latents.dtype
            )  # (seq_len_i, 4096)
            prompt_seq_len = self._prompt_seq_lens[idx]
        else:
            prompt_emb = torch.zeros(1, model.text_dim, device=device, dtype=vae_latents.dtype)
            prompt_seq_len = 1

        # Prepare input as list of per-sample tensors (WanModel interface)
        x_list = [vae_latents[i] for i in range(B)]

        # ---- Inline forward to avoid no_grad in FeatureExtractorWanModel ----
        # Patch embedding
        x_embedded = [model.patch_embedding(u.unsqueeze(0)) for u in x_list]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_embedded]
        )
        x_flat = [u.flatten(2).transpose(1, 2) for u in x_embedded]
        seq_lens = torch.tensor([u.size(1) for u in x_flat], dtype=torch.long)

        # Compute padded sequence length for positional encoding
        max_seq = max(u.size(1) for u in x_flat)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_seq - u.size(1), u.size(2))], dim=1)
            for u in x_flat
        ])

        # Time embedding
        with amp.autocast(dtype=torch.float32):
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t.to(x.dtype)).float()
            )
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        # Context embedding — pad prompt to model.text_len and replicate for batch
        padded_prompt = torch.cat([
            prompt_emb,
            prompt_emb.new_zeros(model.text_len - prompt_emb.size(0), prompt_emb.size(1))
        ])  # (text_len, 4096)
        context_encoded = model.text_embedding(
            padded_prompt.unsqueeze(0).expand(B, -1, -1)
        )
        context_lens_val = torch.tensor(
            [prompt_seq_len] * B, device=device
        ) if self._prompt_embeddings is not None else None

        # Prepare RoPE frequencies
        freqs = model.freqs.to(device, dtype=torch.complex128)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context_encoded,
            context_lens=context_lens_val,
        )

        # Run through transformer blocks (LoRA gradients flow here)
        feature_layers = self.hparams.get("dit_feature_layers", None)
        num_blocks = len(model.blocks)

        if self._multi_layer and feature_layers is not None:
            # Normalize negative indices
            target_indices = set()
            for idx in feature_layers:
                target_indices.add(idx if idx >= 0 else num_blocks + idx)

            collected = {}
            for i, block in enumerate(model.blocks):
                x = block(x, **kwargs)
                if i in target_indices:
                    collected[i] = x.clone()

            # Scalar mix: weighted sum of layer outputs
            normed_weights = torch.softmax(self.layer_weights, dim=0)
            sorted_keys = sorted(collected.keys())
            mixed = sum(
                normed_weights[j] * collected[k] for j, k in enumerate(sorted_keys)
            )
            x = self.layer_gamma * mixed
        else:
            for i, block in enumerate(model.blocks):
                x = block(x, **kwargs)

        # Extract temporal features: reshape [B, L, dim] -> [B, dim, T', H', W']
        temporal_features = []
        for b in range(B):
            T_g, H_g, W_g = grid_sizes[b].tolist()
            feat = x[b, : T_g * H_g * W_g].view(T_g, H_g, W_g, -1)  # (T', H', W', dim)
            feat = feat.permute(3, 0, 1, 2)  # (dim, T', H', W')
            temporal_features.append(feat)

        return torch.stack(temporal_features)  # (B, dim, T', H', W')

    # ------------------------------------------------------------------ #
    #  Loss                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def coral_loss(logits, levels, importance_weights=None, reduction="mean",
                   label_smoothing=0.0, distance_penalty=False, focal_gamma=None):
        """CORAL ordinal loss (reused from MultimodalMultiTaskCoralClassifier)."""
        return MultimodalMultiTaskCoralClassifier.coral_loss(
            logits, levels,
            importance_weights=importance_weights,
            reduction=reduction,
            label_smoothing=label_smoothing,
            distance_penalty=distance_penalty,
            focal_gamma=focal_gamma,
        )

    @staticmethod
    def prob_to_label(probs, num_classes=None):
        labels = torch.sum(probs > 0.5, dim=1)
        if num_classes is not None:
            labels = torch.clamp(labels, max=num_classes - 1)
        return labels

    # ------------------------------------------------------------------ #
    #  Metric updates                                                     #
    # ------------------------------------------------------------------ #

    def _logits_to_preds(self, logits, num_classes, head_type):
        """Convert logits to predicted labels based on head type."""
        if head_type == "ce":
            return logits.argmax(dim=1)
        else:
            return self.prob_to_label(torch.sigmoid(logits), num_classes)

    def _update_metrics(self, out, pain_labels, stim_labels, stage):
        eval_head = self.hparams.get("eval_head", "coral")
        pain_key = f"pain_{eval_head}" if f"pain_{eval_head}" in out else "pain_coral"
        stim_key = f"stim_{eval_head}" if f"stim_{eval_head}" in out else "stim_coral"
        head_type = eval_head if pain_key.endswith(eval_head) else "coral"

        valid_pain = pain_labels != -1
        if valid_pain.any():
            p_logits = out[pain_key][valid_pain]
            p_labels = pain_labels[valid_pain]
            p_preds = self._logits_to_preds(p_logits, self.hparams.num_pain_classes, head_type)
            getattr(self, f"{stage}_pain_mae").update(p_preds, p_labels)
            getattr(self, f"{stage}_pain_qwk").update(p_preds, p_labels)
            getattr(self, f"{stage}_pain_acc").update(p_preds, p_labels)
            getattr(self, f"{stage}_pain_f1").update(p_preds, p_labels)
            if stage in ("val", "test"):
                getattr(self, f"{stage}_pain_recall_per_class").update(p_preds, p_labels)
                getattr(self, f"{stage}_pain_cm").update(p_preds, p_labels)

        valid_stim = stim_labels != -1
        if valid_stim.any() and stim_key in out:
            s_logits = out[stim_key][valid_stim]
            s_labels = stim_labels[valid_stim]
            s_preds = self._logits_to_preds(s_logits, self.hparams.num_stimulus_classes, head_type)
            getattr(self, f"{stage}_stim_mae").update(s_preds, s_labels)
            getattr(self, f"{stage}_stim_qwk").update(s_preds, s_labels)
            getattr(self, f"{stage}_stim_acc").update(s_preds, s_labels)
            getattr(self, f"{stage}_stim_f1").update(s_preds, s_labels)
            if stage in ("val", "test"):
                getattr(self, f"{stage}_stim_recall_per_class").update(s_preds, s_labels)
                getattr(self, f"{stage}_stim_cm").update(s_preds, s_labels)

    # ------------------------------------------------------------------ #
    #  Training / Validation / Test steps                                 #
    # ------------------------------------------------------------------ #

    def _compute_ce_loss(self, ce_logits, labels):
        """Compute CE loss with optional focal weighting."""
        gamma = self.hparams.get("ce_focal_gamma", None)
        ls = self.hparams.get("ce_label_smoothing", 0.0)
        if gamma and gamma > 0.0:
            ce_per = F.cross_entropy(ce_logits, labels, label_smoothing=ls, reduction="none")
            with torch.no_grad():
                pt = torch.softmax(ce_logits, dim=1).gather(
                    1, labels.unsqueeze(1)
                ).squeeze(1).clamp_min(1e-8)
            focal_w = (1.0 - pt) ** gamma
            return (focal_w * ce_per).mean()
        return F.cross_entropy(ce_logits, labels, label_smoothing=ls)

    def _compute_task_loss(self, out, pain_labels, stim_labels):
        """Compute total loss for a batch given model outputs and labels."""
        coral_alpha = self.hparams.get("coral_alpha", 1.0)
        ce_alpha = self.hparams.get("ce_alpha", 0.0)

        valid_pain = pain_labels != -1
        if valid_pain.any() and coral_alpha > 0:
            pain_coral_loss = self.coral_loss(
                out["pain_coral"][valid_pain], pain_labels[valid_pain],
                label_smoothing=self.hparams.label_smoothing,
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma,
            )
        else:
            pain_coral_loss = torch.tensor(0.0, device=self.device)

        if valid_pain.any() and ce_alpha > 0 and self._use_ce:
            pain_ce_loss = self._compute_ce_loss(
                out["pain_ce"][valid_pain], pain_labels[valid_pain]
            )
        else:
            pain_ce_loss = torch.tensor(0.0, device=self.device)

        pain_loss = coral_alpha * pain_coral_loss + ce_alpha * pain_ce_loss

        valid_stim = stim_labels != -1
        if valid_stim.any() and coral_alpha > 0 and "stim_coral" in out:
            stim_coral_loss = self.coral_loss(
                out["stim_coral"][valid_stim], stim_labels[valid_stim],
                label_smoothing=0.0,
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma,
            )
        else:
            stim_coral_loss = torch.tensor(0.0, device=self.device)

        if valid_stim.any() and ce_alpha > 0 and self._use_ce and "stim_ce" in out:
            stim_ce_loss = self._compute_ce_loss(
                out["stim_ce"][valid_stim], stim_labels[valid_stim]
            )
        else:
            stim_ce_loss = torch.tensor(0.0, device=self.device)

        stim_loss = coral_alpha * stim_coral_loss + ce_alpha * stim_ce_loss

        total_loss = (
            self.hparams.pain_loss_weight * pain_loss
            + self.hparams.stim_loss_weight * stim_loss
        )
        return total_loss, pain_loss, stim_loss, pain_coral_loss, pain_ce_loss

    def training_step(self, batch, batch_idx):
        video, pain_labels, stim_labels = batch

        mixup_alpha = self.hparams.get("mixup_alpha", 0.0)

        if mixup_alpha > 0.0 and self.training:
            # Video-level MixUp: blend two samples to disrupt identity cues.
            # Only mix samples that both have valid labels (>= 0).
            valid = (pain_labels >= 0)
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            B = video.size(0)
            perm = torch.randperm(B, device=video.device)

            # Fall back to standard training if permuted labels contain invalids
            both_valid = valid & (pain_labels[perm] >= 0)
            if both_valid.all():
                video_mixed = lam * video + (1.0 - lam) * video[perm]
            else:
                video_mixed = video
                lam = 1.0
            out = self(video_mixed)

            loss_a, pain_loss_a, stim_loss_a, pain_coral_a, pain_ce_a = \
                self._compute_task_loss(out, pain_labels, stim_labels)
            loss_b, pain_loss_b, stim_loss_b, pain_coral_b, pain_ce_b = \
                self._compute_task_loss(out, pain_labels[perm], stim_labels[perm])

            total_loss = lam * loss_a + (1.0 - lam) * loss_b
            pain_loss = lam * pain_loss_a + (1.0 - lam) * pain_loss_b
            stim_loss = lam * stim_loss_a + (1.0 - lam) * stim_loss_b
            pain_coral_loss = lam * pain_coral_a + (1.0 - lam) * pain_coral_b
            pain_ce_loss = lam * pain_ce_a + (1.0 - lam) * pain_ce_b

            # Metrics use primary labels (label_a) for monitoring consistency
            self._update_metrics(out, pain_labels, stim_labels, "train")
        else:
            out = self(video)
            total_loss, pain_loss, stim_loss, pain_coral_loss, pain_ce_loss = \
                self._compute_task_loss(out, pain_labels, stim_labels)
            self._update_metrics(out, pain_labels, stim_labels, "train")

        # Loss breakdown
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_pain_loss", pain_loss, on_step=False, on_epoch=True)
        self.log("train_stim_loss", stim_loss, on_step=False, on_epoch=True)
        if self._use_ce:
            self.log("train_pain_coral_loss", pain_coral_loss, on_step=False, on_epoch=True)
            self.log("train_pain_ce_loss", pain_ce_loss, on_step=False, on_epoch=True)

        # Core metrics
        self.log("train_pain_QWK", self.train_pain_qwk, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pain_MAE", self.train_pain_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pain_Accuracy", self.train_pain_acc, on_step=False, on_epoch=True)
        self.log("train_pain_F1", self.train_pain_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_stim_QWK", self.train_stim_qwk, on_step=False, on_epoch=True)
        self.log("train_stim_MAE", self.train_stim_mae, on_step=False, on_epoch=True)
        self.log("train_stim_Accuracy", self.train_stim_acc, on_step=False, on_epoch=True)
        self.log("train_stim_F1", self.train_stim_f1, on_step=False, on_epoch=True)

        # Learning rate tracking
        opt = self.optimizers()
        if opt is not None:
            for i, pg in enumerate(opt.param_groups):
                self.log(f"lr_group_{i}", pg["lr"], on_step=False, on_epoch=True, prog_bar=(i == 0))

        return total_loss

    def validation_step(self, batch, batch_idx):
        video, pain_labels, stim_labels = batch
        out = self(video)
        self._update_metrics(out, pain_labels, stim_labels, "val")

        coral_alpha = self.hparams.get("coral_alpha", 1.0)
        ce_alpha = self.hparams.get("ce_alpha", 0.0)

        valid_pain = pain_labels != -1
        valid_stim = stim_labels != -1
        pain_coral = self.coral_loss(out["pain_coral"][valid_pain], pain_labels[valid_pain]) if valid_pain.any() else torch.tensor(0.0, device=self.device)
        stim_coral = self.coral_loss(out["stim_coral"][valid_stim], stim_labels[valid_stim]) if (valid_stim.any() and "stim_coral" in out) else torch.tensor(0.0, device=self.device)
        pain_ce = self._compute_ce_loss(out["pain_ce"][valid_pain], pain_labels[valid_pain]) if (valid_pain.any() and self._use_ce) else torch.tensor(0.0, device=self.device)
        stim_ce = self._compute_ce_loss(out["stim_ce"][valid_stim], stim_labels[valid_stim]) if (valid_stim.any() and self._use_ce and "stim_ce" in out) else torch.tensor(0.0, device=self.device)
        pain_loss = coral_alpha * pain_coral + ce_alpha * pain_ce
        stim_loss = coral_alpha * stim_coral + ce_alpha * stim_ce
        val_loss = self.hparams.pain_loss_weight * pain_loss + self.hparams.stim_loss_weight * stim_loss

        sd = {"sync_dist": True}
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, **sd)
        self.log("val_pain_loss", pain_loss, on_step=False, on_epoch=True, **sd)
        self.log("val_stim_loss", stim_loss, on_step=False, on_epoch=True, **sd)
        if self._use_ce:
            self.log("val_pain_coral_loss", pain_coral, on_step=False, on_epoch=True, **sd)
            self.log("val_pain_ce_loss", pain_ce, on_step=False, on_epoch=True, **sd)

        # Core metrics
        self.log("val_pain_QWK", self.val_pain_qwk, on_step=False, on_epoch=True, prog_bar=True, **sd)
        self.log("val_pain_MAE", self.val_pain_mae, on_step=False, on_epoch=True, prog_bar=True, **sd)
        self.log("val_pain_Accuracy", self.val_pain_acc, on_step=False, on_epoch=True, prog_bar=True, **sd)
        self.log("val_pain_F1", self.val_pain_f1, on_step=False, on_epoch=True, prog_bar=True, **sd)
        self.log("val_stim_QWK", self.val_stim_qwk, on_step=False, on_epoch=True, **sd)
        self.log("val_stim_MAE", self.val_stim_mae, on_step=False, on_epoch=True, **sd)
        self.log("val_stim_Accuracy", self.val_stim_acc, on_step=False, on_epoch=True, **sd)
        self.log("val_stim_F1", self.val_stim_f1, on_step=False, on_epoch=True, **sd)

    def on_validation_epoch_end(self):
        """Log per-class recall, prediction distribution, and print summary."""
        if self.trainer.sanity_checking:
            return
        n_pain = self.hparams.num_pain_classes
        recall = self.val_pain_recall_per_class.compute()
        cm = self.val_pain_cm.compute()  # (K, K) — rows=true, cols=pred
        pred_counts = cm.sum(dim=0).long()  # predictions per class
        true_counts = cm.sum(dim=1).long()  # ground truth per class

        for i in range(n_pain):
            self.log(f"val_pain_recall_c{i}", float(recall[i]),
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val_pain_pred_count_c{i}", float(pred_counts[i]),
                     on_step=False, on_epoch=True, sync_dist=True)

        print(f"[Epoch {self.current_epoch}] val per-class recall: "
              + " | ".join(f"c{i}={recall[i]:.3f}" for i in range(n_pain)))
        print(f"[Epoch {self.current_epoch}] val pred distribution: "
              + " | ".join(f"c{i}={pred_counts[i]}" for i in range(n_pain))
              + f"  (true: {' | '.join(f'c{i}={true_counts[i]}' for i in range(n_pain))})")

    def test_step(self, batch, batch_idx):
        video, pain_labels, stim_labels = batch
        out = self(video)
        self._update_metrics(out, pain_labels, stim_labels, "test")
        sd = {"sync_dist": True}
        self.log("test_pain_QWK", self.test_pain_qwk, on_step=False, on_epoch=True, **sd)
        self.log("test_pain_MAE", self.test_pain_mae, on_step=False, on_epoch=True, **sd)
        self.log("test_pain_Accuracy", self.test_pain_acc, on_step=False, on_epoch=True, **sd)
        self.log("test_pain_F1", self.test_pain_f1, on_step=False, on_epoch=True, **sd)
        self.log("test_stim_QWK", self.test_stim_qwk, on_step=False, on_epoch=True, **sd)
        self.log("test_stim_MAE", self.test_stim_mae, on_step=False, on_epoch=True, **sd)
        self.log("test_stim_Accuracy", self.test_stim_acc, on_step=False, on_epoch=True, **sd)
        self.log("test_stim_F1", self.test_stim_f1, on_step=False, on_epoch=True, **sd)

    def on_test_epoch_end(self):
        """Log per-class recall, prediction distribution, and confusion matrix."""
        n_pain = self.hparams.num_pain_classes
        recall = self.test_pain_recall_per_class.compute()
        cm = self.test_pain_cm.compute()  # (K, K) — rows=true, cols=pred
        pred_counts = cm.sum(dim=0).long()
        true_counts = cm.sum(dim=1).long()

        for i in range(n_pain):
            self.log(f"test_pain_recall_c{i}", float(recall[i]), sync_dist=True)
            self.log(f"test_pain_pred_count_c{i}", float(pred_counts[i]), sync_dist=True)

        print(f"[Test] per-class recall: "
              + " | ".join(f"c{i}={recall[i]:.3f}" for i in range(n_pain)))
        print(f"[Test] pred distribution: "
              + " | ".join(f"c{i}={pred_counts[i]}" for i in range(n_pain))
              + f"  (true: {' | '.join(f'c{i}={true_counts[i]}' for i in range(n_pain))})")

        # Full confusion matrix (rows=true label, cols=predicted)
        class_names = [f"c{i}" for i in range(n_pain)]
        header = "        " + "  ".join(f"{c:>6s}" for c in class_names)
        print(f"\n[Test] Confusion Matrix (rows=true, cols=pred):")
        print(header)
        for i in range(n_pain):
            row = "  ".join(f"{int(cm[i, j]):6d}" for j in range(n_pain))
            print(f"  {class_names[i]:>4s}  {row}")

    # ------------------------------------------------------------------ #
    #  Optimizer (separate LR for LoRA vs head)                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        lora_params = [
            p for n, p in self.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad
        ]
        head_params = [
            p for n, p in self.named_parameters()
            if ("lora_A" not in n and "lora_B" not in n) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.hparams.lr_backbone},
                {"params": head_params, "lr": self.hparams.lr_head},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------ #
    #  LoRA checkpoint helpers                                            #
    # ------------------------------------------------------------------ #

    def save_lora_only(self, path: str) -> None:
        """Save only LoRA weights from the DiT backbone."""
        save_lora_weights(self.feature_extractor.model, path)

    def on_train_start(self):
        """Log trainable parameter summary at the start of training."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[OnlineMultimodalClassifier] Total params: {total:,} | Trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0 and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            self.log("vram_peak_alloc_mb", peak_mb, on_step=False, on_epoch=True)
            self.log("vram_peak_reserved_mb", reserved_mb, on_step=False, on_epoch=True)
            print(f"[VRAM] Peak allocated: {peak_mb:.0f} MB | Peak reserved: {reserved_mb:.0f} MB | "
                  f"Total: {total_mb:.0f} MB | Free headroom: {total_mb - reserved_mb:.0f} MB")


class BioVidOnlineClassifier(OnlineMultimodalClassifier):
    """Single-task variant for BioVid pain classification (5-class ordinal).

    Same architecture as OnlineMultimodalClassifier but simplified for
    single-task training on BioVid pain levels (0-4).

    The batch format is (video_tensor, label) instead of
    (video_tensor, pain_label, stim_label).
    """

    def __init__(self, num_classes: int = 5, **kwargs):
        kwargs.setdefault("num_pain_classes", num_classes)
        kwargs.setdefault("num_stimulus_classes", 2)
        kwargs.setdefault("stim_loss_weight", 0.0)
        super().__init__(**kwargs)
        self.hparams["num_classes"] = num_classes

    def _single_to_multi(self, batch):
        """Convert (video, label) batch to (video, pain_label, stim_label)."""
        video, label = batch
        stim_label = torch.full_like(label, -1)
        return video, label, stim_label

    def training_step(self, batch, batch_idx):
        return super().training_step(self._single_to_multi(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._single_to_multi(batch), batch_idx)

    def test_step(self, batch, batch_idx):
        return super().test_step(self._single_to_multi(batch), batch_idx)
