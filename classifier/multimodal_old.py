import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Union, List, Dict, Optional, Tuple
from torchmetrics.classification import CohenKappa, ConfusionMatrix

class MultiModalFusionClassifier(pl.LightningModule):
    """
    Multi-modal fusion classifier that combines VAE and xDiT features for pain/stimulus classification.
    
    This model supports multiple fusion strategies:
    1. Early fusion: Combines features before encoding
    2. Late fusion: Processes each modality separately and combines before classification
    3. Cross-attention fusion: Uses attention to dynamically combine modality information
    
    Args:
        vae_input_shape: Shape of VAE features [C, T, H, W] (e.g., [16, 5, 16, 16])
        xdit_input_shape: Shape of xDiT features [C, T, H, W] (e.g., [1536, 5, 16, 16])
        num_pain_classes: Number of ordinal classes for pain level task
        num_stimulus_classes: Number of ordinal classes for stimulus level task
        fusion_type: Type of fusion to use ('early', 'late', 'cross_attention')
        encoder_hidden_dims: List of hidden dimensions for the shared MLP encoder
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer ('AdamW', 'Adam')
        pain_loss_weight: Weight for pain task loss
        stim_loss_weight: Weight for stimulus task loss
        weight_decay: Weight decay for optimizer
        label_smoothing: Amount of label smoothing to apply
        use_distance_penalty: Whether to use distance penalty in loss
        focal_gamma: Gamma parameter for focal loss weighting
        class_weights: Optional class weights for weighted loss
        encoder_dropout: Dropout rate for encoder layers
        use_lr_scheduler: Whether to use learning rate scheduler
        lr_factor: Factor to reduce learning rate by
        lr_patience: Patience for learning rate scheduler
        min_lr: Minimum learning rate
        monitor_metric: Metric to monitor for early stopping and LR scheduling
    """
    def __init__(
        self,
        vae_input_shape: List[int],   # [C, T, H, W]
        xdit_input_shape: List[int],  # [C, T, H, W]
        num_pain_classes: int,
        num_stimulus_classes: int,
        fusion_type: str = 'late',     # 'early', 'late', or 'cross_attention'
        encoder_hidden_dims: Union[List[int], None] = None,
        learning_rate: float = 1e-4,
        optimizer_name: str = 'AdamW',
        pain_loss_weight: float = 1.0,
        stim_loss_weight: float = 1.0,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        use_distance_penalty: bool = False,
        focal_gamma: Union[float, None] = None,
        class_weights: Union[torch.Tensor, None] = None,
        encoder_dropout: float = 0.5,
        use_lr_scheduler: bool = False,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        monitor_metric: str = 'val_pain_QWK',
    ):
        super().__init__()
        
        if num_pain_classes <= 1 or num_stimulus_classes <= 1:
            raise ValueError("Number of classes for each task must be >= 2 for CORAL.")
            
        if fusion_type not in ['early', 'late', 'cross_attention']:
            raise ValueError(f"Fusion type must be one of ['early', 'late', 'cross_attention'], got {fusion_type}")

        # Store hyperparameters
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Store class weights for weighted loss calculation
        self.register_buffer('class_weights', class_weights if class_weights is not None 
                            else torch.ones(num_pain_classes))
        
        # Extract dimensions from input shapes
        vae_c, vae_t, vae_h, vae_w = vae_input_shape
        xdit_c, xdit_t, xdit_h, xdit_w = xdit_input_shape
        
        # Verify that temporal and spatial dimensions match
        if vae_t != xdit_t or vae_h != xdit_h or vae_w != xdit_w:
            raise ValueError(f"Temporal and spatial dimensions must match. VAE: {vae_t}x{vae_h}x{vae_w}, xDiT: {xdit_t}x{xdit_h}x{xdit_w}")
        
        # --- Feature Processing Modules ---
        # These modules process each modality before fusion
        self.vae_processor = self._build_feature_processor(vae_c, vae_t, vae_h, vae_w)
        self.xdit_processor = self._build_feature_processor(xdit_c, xdit_t, xdit_h, xdit_w)
        
        # --- Fusion Module ---
        if fusion_type == 'early':
            # Early fusion: concatenate features and then process
            self.fusion_module = self._build_early_fusion(vae_c, xdit_c)
            fusion_output_dim = self._calculate_fusion_output_dim(vae_input_shape, xdit_input_shape, 'early')
        elif fusion_type == 'late':
            # Late fusion: process each modality separately and then combine
            self.fusion_module = self._build_late_fusion(vae_c, xdit_c)
            fusion_output_dim = self._calculate_fusion_output_dim(vae_input_shape, xdit_input_shape, 'late')
        else:  # cross_attention
            # Cross-attention fusion: use attention to combine modalities
            self.fusion_module = self._build_cross_attention_fusion(vae_c, xdit_c)
            fusion_output_dim = self._calculate_fusion_output_dim(vae_input_shape, xdit_input_shape, 'cross_attention')
        
        # --- Shared Encoder (after fusion) ---
        if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
            self.shared_encoder = nn.Sequential(
                nn.Linear(fusion_output_dim, fusion_output_dim),
                nn.BatchNorm1d(fusion_output_dim),
                nn.ReLU(),
                nn.Dropout(encoder_dropout)
            )
            encoder_output_dim = fusion_output_dim
        else:
            layers = []
            current_dim = fusion_output_dim
            for h_dim in encoder_hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(encoder_dropout))
                current_dim = h_dim
            self.shared_encoder = nn.Sequential(*layers)
            encoder_output_dim = current_dim
            
        # --- CORAL Heads ---
        self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
        self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)
        
        # --- Metrics ---
        # Using same metrics as MultiTaskCoralClassifier
        metric_args = {'dist_sync_on_step': False}
        
        # MAE Metrics
        self.train_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_pain_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.train_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.val_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)
        self.test_stim_mae = torchmetrics.MeanAbsoluteError(**metric_args)

        # QWK Metrics
        self.train_pain_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_pain_classes, weights='quadratic', **metric_args)
        self.val_pain_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_pain_classes, weights='quadratic', **metric_args)
        self.test_pain_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_pain_classes, weights='quadratic', **metric_args)
        self.train_stim_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_stimulus_classes, weights='quadratic', **metric_args)
        self.val_stim_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_stimulus_classes, weights='quadratic', **metric_args)
        self.test_stim_qwk = CohenKappa(task="multiclass", num_classes=self.hparams.num_stimulus_classes, weights='quadratic', **metric_args)

        # Accuracy Metrics
        self.train_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        self.val_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        self.test_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        self.train_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)
        self.val_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)
        self.test_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)

        # Confusion Matrix Metrics
        metric_args_cm = {'dist_sync_on_step': False, 'normalize': None}
        if self.hparams.num_pain_classes > 0:
            self.test_pain_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_pain_classes, **metric_args_cm)
        if self.hparams.num_stimulus_classes > 0:
            self.test_stim_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_stimulus_classes, **metric_args_cm)
    
    def _build_feature_processor(self, c: int, t: int, h: int, w: int) -> nn.Module:
        """Build a module to process raw features from each modality."""
        return nn.Sequential(
            # Reshape to [B, C, T*H*W]
            nn.Flatten(start_dim=2),
            # Apply 1D convolution to process temporal features
            nn.Conv1d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            # Reshape back to [B, C, T, H, W]
            nn.Unflatten(2, (t, h, w))
        )
    
    def _build_early_fusion(self, vae_c: int, xdit_c: int) -> nn.Module:
        """Build early fusion module that concatenates features along channel dimension."""
        return nn.Sequential(
            # Concatenate along channel dimension and then project to a common dimension
            nn.Conv3d(vae_c + xdit_c, 512, kernel_size=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Flatten(start_dim=1)  # Flatten to [B, 512*T*H*W]
        )
    
    def _build_late_fusion(self, vae_c: int, xdit_c: int) -> nn.Module:
        """Build late fusion module that processes each modality separately."""
        vae_dim = 256
        xdit_dim = 256
        
        return nn.ModuleDict({
            'vae_encoder': nn.Sequential(
                nn.Conv3d(vae_c, vae_dim, kernel_size=1),
                nn.BatchNorm3d(vae_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),  # Global pooling to [B, vae_dim, 1, 1, 1]
                nn.Flatten(start_dim=1)   # Flatten to [B, vae_dim]
            ),
            'xdit_encoder': nn.Sequential(
                nn.Conv3d(xdit_c, xdit_dim, kernel_size=1),
                nn.BatchNorm3d(xdit_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),  # Global pooling to [B, xdit_dim, 1, 1, 1]
                nn.Flatten(start_dim=1)   # Flatten to [B, xdit_dim]
            ),
            'fusion': nn.Sequential(
                nn.Linear(vae_dim + xdit_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
        })
    
    def _build_cross_attention_fusion(self, vae_c: int, xdit_c: int) -> nn.Module:
        """Build cross-attention fusion module that uses attention to combine modalities."""
        hidden_dim = 512
        num_heads = 8
        
        return nn.ModuleDict({
            'vae_proj': nn.Sequential(
                nn.Conv3d(vae_c, hidden_dim, kernel_size=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Flatten(start_dim=2)  # [B, hidden_dim, T*H*W]
            ),
            'xdit_proj': nn.Sequential(
                nn.Conv3d(xdit_c, hidden_dim, kernel_size=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(),
                nn.Flatten(start_dim=2)  # [B, hidden_dim, T*H*W]
            ),
            'cross_attention': nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            ),
            'output_proj': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        })
    
    def _calculate_fusion_output_dim(self, vae_shape: List[int], xdit_shape: List[int], fusion_type: str) -> int:
        """Calculate the output dimension of the fusion module."""
        vae_c, vae_t, vae_h, vae_w = vae_shape
        xdit_c, xdit_t, xdit_h, xdit_w = xdit_shape
        
        if fusion_type == 'early':
            return 512 * vae_t * vae_h * vae_w
        elif fusion_type == 'late':
            return 512  # Defined in _build_late_fusion
        else:  # cross_attention
            return 512  # Defined in _build_cross_attention_fusion
    
    def forward(self, vae_features: torch.Tensor, xdit_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            vae_features: VAE features [B, C, T, H, W]
            xdit_features: xDiT features [B, C, T, H, W]
            
        Returns:
            tuple: (pain_logits, stimulus_logits)
        """
        # Process each modality
        vae_processed = self.vae_processor(vae_features)
        xdit_processed = self.xdit_processor(xdit_features)
        
        # Apply fusion based on fusion type
        if self.hparams.fusion_type == 'early':
            # Concatenate along channel dimension
            fused = torch.cat([vae_processed, xdit_processed], dim=1)
            # Apply fusion module
            fused = self.fusion_module(fused)
        
        elif self.hparams.fusion_type == 'late':
            # Process each modality separately
            vae_encoded = self.fusion_module['vae_encoder'](vae_processed)
            xdit_encoded = self.fusion_module['xdit_encoder'](xdit_processed)
            # Concatenate encoded features
            concat_features = torch.cat([vae_encoded, xdit_encoded], dim=1)
            # Apply fusion
            fused = self.fusion_module['fusion'](concat_features)
        
        else:  # cross_attention
            # Project each modality
            vae_proj = self.fusion_module['vae_proj'](vae_processed)  # [B, hidden_dim, T*H*W]
            xdit_proj = self.fusion_module['xdit_proj'](xdit_processed)  # [B, hidden_dim, T*H*W]
            
            # Transpose for attention [B, T*H*W, hidden_dim]
            vae_proj = vae_proj.transpose(1, 2)
            xdit_proj = xdit_proj.transpose(1, 2)
            
            # Apply cross-attention (vae attends to xdit)
            attn_output, _ = self.fusion_module['cross_attention'](
                query=vae_proj,
                key=xdit_proj,
                value=xdit_proj
            )
            
            # Global pooling
            fused = attn_output.mean(dim=1)  # [B, hidden_dim]
            
            # Final projection
            fused = self.fusion_module['output_proj'](fused)
        
        # Apply shared encoder
        encoded = self.shared_encoder(fused)
        
        # Apply task-specific heads
        pain_logits = self.pain_head(encoded)
        stimulus_logits = self.stimulus_head(encoded)
        
        return pain_logits, stimulus_logits
    
    @staticmethod
    def coral_loss(logits, levels, importance_weights=None, reduction='mean', label_smoothing=0.0, distance_penalty=False, focal_gamma=None):
        """
        Compute the CORAL loss for ordinal regression.
        Identical to the implementation in MultiTaskCoralClassifier.
        """
        if logits.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        num_classes_minus1 = logits.shape[1]
        levels = levels.long()

        # Step 1: Generate target binary levels
        levels_binary = (levels.unsqueeze(1) > torch.arange(num_classes_minus1, device=logits.device).unsqueeze(0)).float()

        if label_smoothing > 0.0:
            levels_binary = levels_binary * (1 - label_smoothing) + label_smoothing / 2

        # Step 2: Base CORAL loss computation
        log_sigmoid = F.logsigmoid(logits)
        base_loss_tasks = log_sigmoid * levels_binary + (log_sigmoid - logits) * (1 - levels_binary)

        # Step 3: Apply Distance Penalty (Optional)
        if distance_penalty:
            distance_matrix = torch.abs(
                levels.unsqueeze(1) - torch.arange(num_classes_minus1, device=logits.device).unsqueeze(0)
            ).float()  # (batch_size, num_classes-1)
            base_loss_tasks = base_loss_tasks * (1.0 + distance_matrix)  # Penalize farther threshold mistakes more

        # Step 4: Apply Focal Weighting (Optional)
        if focal_gamma is not None:
            probs = torch.sigmoid(logits)
            eps = 1e-6
            focal_weight = torch.where(
                levels_binary > 0.5,
                (1 - probs + eps) ** focal_gamma,
                (probs + eps) ** focal_gamma
            )
            base_loss_tasks = focal_weight * base_loss_tasks

        # Step 5: Sum across tasks
        loss_per_sample = -torch.sum(base_loss_tasks, dim=1)

        # Step 6: Importance Weights
        if importance_weights is not None:
            loss_per_sample *= importance_weights

        # Step 7: Reduction
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
        """
        Converts predicted probabilities to labels.
        Identical to the implementation in MultiTaskCoralClassifier.
        """
        if probs is None:
            return None
            
        # For CORAL, we predict K-1 binary probabilities
        # The predicted class is the sum of all positive predictions
        preds = (probs >= 0.5).sum(dim=1)
        
        # Ensure predictions are within valid range [0, num_classes-1]
        if num_classes is not None:
            preds = torch.clamp(preds, 0, num_classes - 1)
            
        return preds
    
    def _calculate_loss_and_metrics(self, pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage: str, batch_idx: int = 0):
        """
        Calculate loss and metrics for both tasks.
        Similar to the implementation in MultiTaskCoralClassifier.
        """
        # Initialize total loss
        total_loss = 0.0
        loss_dict = {}
        
        # Process pain task if we have labels
        if pain_labels is not None and not torch.all(pain_labels < 0):
            valid_mask = pain_labels >= 0
            if torch.any(valid_mask):
                # Get valid samples
                valid_pain_logits = pain_logits[valid_mask]
                valid_pain_labels = pain_labels[valid_mask]
                
                # Calculate loss
                pain_loss = self.coral_loss(
                    valid_pain_logits, 
                    valid_pain_labels,
                    label_smoothing=self.hparams.label_smoothing,
                    distance_penalty=self.hparams.use_distance_penalty,
                    focal_gamma=self.hparams.focal_gamma
                )
                
                # Apply task weight
                weighted_pain_loss = pain_loss * self.hparams.pain_loss_weight
                total_loss += weighted_pain_loss
                loss_dict['pain_loss'] = pain_loss.item()
                
                # Calculate metrics
                pain_probs = torch.sigmoid(valid_pain_logits)
                pain_preds = self.prob_to_label(pain_probs, self.hparams.num_pain_classes)
                
                # Update metrics based on stage
                if stage == 'train':
                    self.train_pain_mae(pain_preds.float(), valid_pain_labels.float())
                    self.train_pain_qwk(pain_preds, valid_pain_labels)
                    self.train_pain_acc(pain_preds, valid_pain_labels)
                    self.log('train_pain_loss', pain_loss.item(), prog_bar=True)
                elif stage == 'val':
                    self.val_pain_mae(pain_preds.float(), valid_pain_labels.float())
                    self.val_pain_qwk(pain_preds, valid_pain_labels)
                    self.val_pain_acc(pain_preds, valid_pain_labels)
                    self.log('val_pain_loss', pain_loss.item(), prog_bar=True)
                else:  # test
                    self.test_pain_mae(pain_preds.float(), valid_pain_labels.float())
                    self.test_pain_qwk(pain_preds, valid_pain_labels)
                    self.test_pain_acc(pain_preds, valid_pain_labels)
                    self.test_pain_cm(pain_preds, valid_pain_labels)
                    self.log('test_pain_loss', pain_loss.item(), prog_bar=True)
        
        # Process stimulus task if we have labels
        if stimulus_labels is not None and not torch.all(stimulus_labels < 0):
            valid_mask = stimulus_labels >= 0
            if torch.any(valid_mask):
                # Get valid samples
                valid_stim_logits = stimulus_logits[valid_mask]
                valid_stim_labels = stimulus_labels[valid_mask]
                
                # Calculate loss
                stim_loss = self.coral_loss(
                    valid_stim_logits, 
                    valid_stim_labels,
                    label_smoothing=self.hparams.label_smoothing,
                    distance_penalty=self.hparams.use_distance_penalty,
                    focal_gamma=self.hparams.focal_gamma
                )
                
                # Apply task weight
                weighted_stim_loss = stim_loss * self.hparams.stim_loss_weight
                total_loss += weighted_stim_loss
                loss_dict['stim_loss'] = stim_loss.item()
                
                # Calculate metrics
                stim_probs = torch.sigmoid(valid_stim_logits)
                stim_preds = self.prob_to_label(stim_probs, self.hparams.num_stimulus_classes)
                
                # Update metrics based on stage
                if stage == 'train':
                    self.train_stim_mae(stim_preds.float(), valid_stim_labels.float())
                    self.train_stim_qwk(stim_preds, valid_stim_labels)
                    self.train_stim_acc(stim_preds, valid_stim_labels)
                    self.log('train_stim_loss', stim_loss.item(), prog_bar=True)
                elif stage == 'val':
                    self.val_stim_mae(stim_preds.float(), valid_stim_labels.float())
                    self.val_stim_qwk(stim_preds, valid_stim_labels)
                    self.val_stim_acc(stim_preds, valid_stim_labels)
                    self.log('val_stim_loss', stim_loss.item(), prog_bar=True)
                else:  # test
                    self.test_stim_mae(stim_preds.float(), valid_stim_labels.float())
                    self.test_stim_qwk(stim_preds, valid_stim_labels)
                    self.test_stim_acc(stim_preds, valid_stim_labels)
                    self.test_stim_cm(stim_preds, valid_stim_labels)
                    self.log('test_stim_loss', stim_loss.item(), prog_bar=True)
        
        # Log total loss
        if stage == 'train':
            self.log('train_loss', total_loss.item(), prog_bar=True)
        elif stage == 'val':
            self.log('val_loss', total_loss.item(), prog_bar=True)
        else:  # test
            self.log('test_loss', total_loss.item(), prog_bar=True)
        
        # Log metrics
        if stage == 'train':
            self.log('train_pain_MAE', self.train_pain_mae, prog_bar=True)
            self.log('train_pain_QWK', self.train_pain_qwk, prog_bar=True)
            self.log('train_pain_acc', self.train_pain_acc, prog_bar=True)
            self.log('train_stim_MAE', self.train_stim_mae, prog_bar=True)
            self.log('train_stim_QWK', self.train_stim_qwk, prog_bar=True)
            self.log('train_stim_acc', self.train_stim_acc, prog_bar=True)
        elif stage == 'val':
            self.log('val_pain_MAE', self.val_pain_mae, prog_bar=True)
            self.log('val_pain_QWK', self.val_pain_qwk, prog_bar=True)
            self.log('val_pain_acc', self.val_pain_acc, prog_bar=True)
            self.log('val_stim_MAE', self.val_stim_mae, prog_bar=True)
            self.log('val_stim_QWK', self.val_stim_qwk, prog_bar=True)
            self.log('val_stim_acc', self.val_stim_acc, prog_bar=True)
        else:  # test
            self.log('test_pain_MAE', self.test_pain_mae, prog_bar=True)
            self.log('test_pain_QWK', self.test_pain_qwk, prog_bar=True)
            self.log('test_pain_acc', self.test_pain_acc, prog_bar=True)
            self.log('test_stim_MAE', self.test_stim_mae, prog_bar=True)
            self.log('test_stim_QWK', self.test_stim_qwk, prog_bar=True)
            self.log('test_stim_acc', self.test_stim_acc, prog_bar=True)
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        vae_features, xdit_features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(vae_features, xdit_features)
        loss, _ = self._calculate_loss_and_metrics(
            pain_logits, stimulus_logits, pain_labels, stimulus_labels, 
            stage='train', batch_idx=batch_idx
        )
        # Log learning rate
        self.log("learning_rate", self.hparams.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        vae_features, xdit_features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(vae_features, xdit_features)
        self._calculate_loss_and_metrics(
            pain_logits, stimulus_logits, pain_labels, stimulus_labels, 
            stage='val', batch_idx=batch_idx
        )
        # No loss returned for validation
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        vae_features, xdit_features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(vae_features, xdit_features)
        self._calculate_loss_and_metrics(
            pain_logits, stimulus_logits, pain_labels, stimulus_labels, 
            stage='test', batch_idx=batch_idx
        )
        # No loss returned for test
    
    def sanity_check(self, batch_size: int = 4):
        """Performs a basic check on input/output shapes."""
        print(f"Running sanity check for {self.__class__.__name__}...")
        try:
            # Use self.device which Lightning manages
            device = self.device
            
            # Create dummy inputs for both modalities
            vae_c, vae_t, vae_h, vae_w = self.hparams.vae_input_shape
            xdit_c, xdit_t, xdit_h, xdit_w = self.hparams.xdit_input_shape
            
            dummy_vae_input = torch.randn(batch_size, vae_c, vae_t, vae_h, vae_w, device=device)
            dummy_xdit_input = torch.randn(batch_size, xdit_c, xdit_t, xdit_h, xdit_w, device=device)
            
            self.eval()  # Set model to evaluation mode for the check
            with torch.no_grad():  # No need to track gradients
                pain_logits, stimulus_logits = self(dummy_vae_input, dummy_xdit_input)
            self.train()  # Set back to train mode
            
            # Check output shapes
            expected_pain_shape = (batch_size, self.hparams.num_pain_classes - 1)
            expected_stim_shape = (batch_size, self.hparams.num_stimulus_classes - 1)
            
            assert pain_logits.shape == expected_pain_shape, \
                f"Pain logits shape mismatch: Expected {expected_pain_shape}, Got {pain_logits.shape}"
            assert stimulus_logits.shape == expected_stim_shape, \
                f"Stimulus logits shape mismatch: Expected {expected_stim_shape}, Got {stimulus_logits.shape}"
            
            print(f"Sanity check passed for {self.__class__.__name__}.")
            print(f"  VAE input shape: {dummy_vae_input.shape}")
            print(f"  xDiT input shape: {dummy_xdit_input.shape}")
            print(f"  Pain logits shape: {pain_logits.shape}")
            print(f"  Stimulus logits shape: {stimulus_logits.shape}")
            # Print head weight norms for debug/diagnostics
            print(f"  Pain head weights: {self.pain_head.weight.data.norm():.4f}")
            print(f"  Stim head weights: {self.stimulus_head.weight.data.norm():.4f}")
            
        except Exception as e:
            print(f"Sanity check failed for {self.__class__.__name__}: {e}")
            raise  # Re-raise the exception after printing
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use."""
        # Explicitly cast learning rate to float
        try:
            lr = float(self.hparams.learning_rate)
            wd = float(self.hparams.weight_decay)
        except ValueError:
            print(f"Error: Could not convert learning rate '{self.hparams.learning_rate}' to float!")
            raise  # Re-raise the error
        
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        
        # Check if lr_scheduler parameters are available in hparams
        if self.hparams.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max' if 'QWK' in self.hparams.monitor_metric else 'min',
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.min_lr
            )
            
            # PyTorch Lightning expects this format for ReduceLROnPlateau
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                }
            }
        
        return optimizer


class MultiModalDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for datasets that provides both VAE and xDiT features.
    
    This wrapper expects a dataset that returns features, pain_label, stimulus_label
    and transforms it to return vae_features, xdit_features, pain_label, stimulus_label.
    
    Args:
        original_dataset: The original dataset to wrap
        vae_features_dir: Directory containing VAE feature files
        xdit_features_dir: Directory containing xDiT feature files
        file_id_mapping_func: Function to map dataset item to feature file ID
    """
    def __init__(
        self,
        original_dataset,
        vae_features_dir: str,
        xdit_features_dir: str,
        file_id_mapping_func=None
    ):
        self.original_dataset = original_dataset
        self.vae_features_dir = vae_features_dir
        self.xdit_features_dir = xdit_features_dir
        self.file_id_mapping_func = file_id_mapping_func or (lambda x: x[3])  # Default to using video_id
        
        # Cache for loaded features
        self.vae_features_cache = {}
        self.xdit_features_cache = {}
        self.cache_size_limit = 100  # Limit cache size to avoid memory issues
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # Get original item
        item = self.original_dataset[idx]
        
        # Extract pain_label and stimulus_label
        if len(item) == 3:  # features, pain_label, stimulus_label
            _, pain_label, stimulus_label = item
        elif len(item) >= 5:  # features, pain_label, class_label, video_id, clip_id
            _, pain_label, stimulus_label = item[0], item[1], item[2]
        else:
            raise ValueError(f"Unexpected item format: {item}")
        
        # Get file ID for loading features
        file_id = self.file_id_mapping_func(item)
        
        # Load VAE features
        if file_id in self.vae_features_cache:
            vae_features = self.vae_features_cache[file_id]
        else:
            vae_path = os.path.join(self.vae_features_dir, f"{file_id}.npy")
            if not os.path.exists(vae_path):
                raise FileNotFoundError(f"VAE features not found at {vae_path}")
            vae_features = torch.from_numpy(np.load(vae_path)).float()
            
            # Update cache
            if len(self.vae_features_cache) < self.cache_size_limit:
                self.vae_features_cache[file_id] = vae_features
        
        # Load xDiT features
        if file_id in self.xdit_features_cache:
            xdit_features = self.xdit_features_cache[file_id]
        else:
            xdit_path = os.path.join(self.xdit_features_dir, f"{file_id}.npy")
            if not os.path.exists(xdit_path):
                raise FileNotFoundError(f"xDiT features not found at {xdit_path}")
            xdit_features = torch.from_numpy(np.load(xdit_path)).float()
            
            # Update cache
            if len(self.xdit_features_cache) < self.cache_size_limit:
                self.xdit_features_cache[file_id] = xdit_features
        
        return vae_features, xdit_features, pain_label, stimulus_label


# Example configuration for multimodal fusion model
def get_default_config():
    """
    Returns default configuration for MultiModalFusionClassifier.
    """
    return {
        "model_params": {
            "vae_input_shape": [16, 5, 16, 16],     # [C, T, H, W] for VAE features
            "xdit_input_shape": [1536, 5, 16, 16],  # [C, T, H, W] for xDiT features
            "num_pain_classes": 5,
            "num_stimulus_classes": 5,
            "fusion_type": "late",                  # 'early', 'late', or 'cross_attention'
            "encoder_hidden_dims": [512, 256],
            "encoder_dropout": 0.5,
            "pain_loss_weight": 1.0,
            "stim_loss_weight": 1.0,
            "label_smoothing": 0.0,
            "use_distance_penalty": False,
            "focal_gamma": None
        },
        "optimizer_params": {
            "optimizer_name": "AdamW",
            "learning_rate": 5.0e-5,
            "weight_decay": 0.01,
            "use_lr_scheduler": True,
            "lr_factor": 0.5,
            "lr_patience": 5,
            "min_lr": 1.0e-6,
            "monitor_metric": "val_pain_QWK"
        },
        "data_params": {
            "vae_features_dir": "/path/to/vae_features",
            "xdit_features_dir": "/path/to/xdit_features",
            "batch_size": 32,
            "num_workers": 4
        }
    }


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Create a sample config
    config = get_default_config()
    
    # Save config to YAML file
    with open("config_multimodal_fusion.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created sample config file: config_multimodal_fusion.yaml")
    
    # Create model instance for sanity check
    model = MultiModalFusionClassifier(
        vae_input_shape=config["model_params"]["vae_input_shape"],
        xdit_input_shape=config["model_params"]["xdit_input_shape"],
        num_pain_classes=config["model_params"]["num_pain_classes"],
        num_stimulus_classes=config["model_params"]["num_stimulus_classes"],
        fusion_type=config["model_params"]["fusion_type"],
        encoder_hidden_dims=config["model_params"]["encoder_hidden_dims"],
        encoder_dropout=config["model_params"]["encoder_dropout"]
    )
    
    # Print model summary
    print(model)
    
    # Run sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.sanity_check(batch_size=2)