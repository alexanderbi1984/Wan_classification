import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim
from typing import Union, List
from torchmetrics.classification import CohenKappa, ConfusionMatrix

# --------------------------------------------------------------------
# Multi-Task CORAL Lightning Module
# --------------------------------------------------------------------

class MultiTaskCoralClassifier(pl.LightningModule):
    """
    Multi-task LightningModule using a shared encoder and two separate CORAL heads.
    Handles potentially missing labels for each task during training/evaluation.

    Args:
        input_dim (int): Dimension of the input features (e.g., 768).
        num_pain_classes (int): Number of ordinal classes for the pain level task (e.g., 5).
        num_stimulus_classes (int): Number of ordinal classes for the stimulus level task (e.g., 5).
        encoder_hidden_dims (list[int], optional): List of hidden dimensions for the shared MLP encoder. Defaults to None (Linear encoder).
        learning_rate (float, optional): Learning rate. Defaults to 1e-4.
        optimizer_name (str, optional): Optimizer name ('AdamW', 'Adam'). Defaults to 'AdamW'.
        pain_loss_weight (float, optional): Weight for the pain task loss. Defaults to 1.0.
        stim_loss_weight (float, optional): Weight for the stimulus task loss. Defaults to 1.0.
        distributed (bool, optional): Flag indicating if the model is distributed. Defaults to False.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.0.
        label_smoothing (float, optional): Amount of smoothing to apply to binary targets. Defaults to 0.0.
        use_distance_penalty (bool, optional): Flag indicating whether to use distance penalty. Defaults to False.
        focal_gamma (float, optional): Focal gamma for focal weighting. Defaults to None.
        class_weights (torch.Tensor, optional): Class weights for weighted loss calculation. Defaults to None.
        encoder_dropout (float, optional): Dropout rate for all encoder dropout layers. Defaults to 0.5.
        # Learning rate scheduler parameters
        use_lr_scheduler: bool = False,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        monitor_metric: str = 'val_pain_QWK',
    """
    def __init__(
        self,
        input_dim: int,
        num_pain_classes: int,
        num_stimulus_classes: int,
        encoder_hidden_dims: Union[List[int], None] = None,
        learning_rate: float = 1e-4,
        optimizer_name: str = 'AdamW',
        pain_loss_weight: float = 1.0,
        stim_loss_weight: float = 1.0,
        distributed: bool = False,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        use_distance_penalty: bool = False,
        focal_gamma: Union[float, None] = None,
        class_weights: Union[torch.Tensor, None] = None,
        encoder_dropout: float = 0.5, # Dropout rate for all encoder dropout layers
        # Learning rate scheduler parameters
        use_lr_scheduler: bool = False,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        min_lr: float = 1e-6,
        monitor_metric: str = 'val_pain_QWK',
    ):
        super().__init__()

        if num_pain_classes <= 1 or num_stimulus_classes <= 1:
            raise ValueError("Number of classes for each task must be >= 2 for CORAL.")

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"Label smoothing must be in [0, 1), got {label_smoothing}")

        # Store hyperparameters (distributed is often not logged, but store attribute)
        self.save_hyperparameters(ignore=['distributed', 'class_weights']) # Don't save weights in checkpoint
        self.distributed = distributed # Store the attribute

        # Store class weights for weighted loss calculation
        self.register_buffer('class_weights', class_weights if class_weights is not None 
                            else torch.ones(num_pain_classes))

        # --- Shared Encoder ---
        if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Dropout(encoder_dropout)
            )
            encoder_output_dim = input_dim
        else:
            layers = []
            current_dim = input_dim
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
        # Using MAE as it's common for ordinal tasks, but others like Accuracy could be added
        metric_args = {'dist_sync_on_step': False} # Manage sync manually if needed or rely on Lightning
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

        # --- Accuracy Metrics ---
        self.train_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        self.val_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        self.test_pain_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_pain_classes, average='macro', **metric_args)
        
        self.train_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)
        self.val_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)
        self.test_stim_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_stimulus_classes, average='macro', **metric_args)

        # --- Confusion Matrix Metrics (for final fold evaluation) ---
        # Using normalize=None for raw counts. Will be computed manually after trainer.test()
        metric_args_cm = {'dist_sync_on_step': False, 'normalize': None} 
        if self.hparams.num_pain_classes > 0:
            self.test_pain_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_pain_classes, **metric_args_cm)
        if self.hparams.num_stimulus_classes > 0:
            self.test_stim_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_stimulus_classes, **metric_args_cm)

    @staticmethod
    def coral_loss(logits, levels, importance_weights=None, reduction='mean', label_smoothing=0.0, distance_penalty=False, focal_gamma=None):
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
        Converts predicted probabilities to labels (moved inside the class).
        Args:
            probs: Predicted probabilities of shape (batch_size, num_classes - 1).
            num_classes: (int, optional) Number of classes for result clamping. If None, no clamp applied.
        Returns:
            Predicted labels of shape (batch_size), clamped to [0, num_classes-1] if num_classes given.
        """
        labels = torch.sum(probs > 0.5, dim=1)
        if num_classes is not None:
            labels = torch.clamp(labels, max=num_classes - 1)
        return labels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the encoder and heads. """
        encoded_features = self.shared_encoder(x)
        pain_logits = self.pain_head(encoded_features)
        stimulus_logits = self.stimulus_head(encoded_features)
        return pain_logits, stimulus_logits

    def _calculate_loss_and_metrics(self, pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage: str, batch_idx: int = 0):
        """ Helper to calculate loss and update metrics for a given stage. """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Ensure loss is on correct device and requires grad
        pain_loss = torch.tensor(0.0, device=self.device)
        stim_loss = torch.tensor(0.0, device=self.device)

        # --- Pain Task ---
        valid_pain_mask = pain_labels != -1
        if valid_pain_mask.any():
            valid_pain_logits = pain_logits[valid_pain_mask]
            valid_pain_labels = pain_labels[valid_pain_mask]
            
            # Get sample weights from class weights if available (for handling class imbalance)
            importance_weights = None
            if hasattr(self, 'class_weights') and self.class_weights is not None and stage == 'train':
                # Map each label to its corresponding class weight
                importance_weights = self.class_weights[valid_pain_labels]
            
            # Only apply label smoothing during training
            smoothing = self.hparams.label_smoothing if stage == 'train' else 0.0
            pain_loss = self.coral_loss(
                valid_pain_logits, 
                valid_pain_labels, 
                importance_weights=importance_weights,
                label_smoothing=smoothing,
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma
            )

            # Update Metrics
            pain_probs = torch.sigmoid(valid_pain_logits)
            pain_preds = self.prob_to_label(pain_probs, num_classes=self.hparams.num_pain_classes)
            mae_metric = getattr(self, f"{stage}_pain_mae")
            mae_metric.update(pain_preds, valid_pain_labels)
            qwk_metric = getattr(self, f"{stage}_pain_qwk")
            # Only update QWK if more than one class in batch
            if len(valid_pain_labels.unique()) > 1:
                qwk_metric.update(pain_preds, valid_pain_labels)
            
            # Update Accuracy for Pain
            acc_metric_pain = getattr(self, f"{stage}_pain_acc")
            acc_metric_pain.update(pain_preds, valid_pain_labels)
            
            # Update Test Confusion Matrix for Pain if stage is 'test'
            if stage == 'test' and hasattr(self, 'test_pain_cm'):
                self.test_pain_cm.update(pain_preds, valid_pain_labels)

            self.log(f"{stage}_pain_loss", pain_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{stage}_pain_MAE", mae_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{stage}_pain_QWK", qwk_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{stage}_pain_Accuracy", acc_metric_pain, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # --- Stimulus Task ---
        valid_stim_mask = stimulus_labels != -1
        if valid_stim_mask.any():
            valid_stim_logits = stimulus_logits[valid_stim_mask]
            valid_stim_labels = stimulus_labels[valid_stim_mask]
            # Use static method via self
            stim_loss = self.coral_loss(
                valid_stim_logits, 
                valid_stim_labels,
                label_smoothing=0.0,  # No smoothing for stimulus task
                distance_penalty=self.hparams.use_distance_penalty,
                focal_gamma=self.hparams.focal_gamma
            )

            # Update Metrics
            stim_probs = torch.sigmoid(valid_stim_logits)
            stim_preds = self.prob_to_label(stim_probs, num_classes=self.hparams.num_stimulus_classes)
            mae_metric = getattr(self, f"{stage}_stim_mae")
            mae_metric.update(stim_preds, valid_stim_labels)
            qwk_metric = getattr(self, f"{stage}_stim_qwk")
            # Only update QWK if more than one class in batch
            if len(valid_stim_labels.unique()) > 1:
                qwk_metric.update(stim_preds, valid_stim_labels)
            
            # Update Accuracy for Stimulus
            acc_metric_stim = getattr(self, f"{stage}_stim_acc")
            acc_metric_stim.update(stim_preds, valid_stim_labels)

            # Update Test Confusion Matrix for Stimulus if stage is 'test'
            if stage == 'test' and hasattr(self, 'test_stim_cm'):
                self.test_stim_cm.update(stim_preds, valid_stim_labels)

            self.log(f"{stage}_stim_loss", stim_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{stage}_stim_MAE", mae_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{stage}_stim_QWK", qwk_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{stage}_stim_Accuracy", acc_metric_stim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # --- Combine Losses ---
        # Only add stim_loss if there are valid stim labels
        # This is more robust for tasks with missing labels in multitask settings
        added_loss = False
        if valid_pain_mask.any():
            total_loss = self.hparams.pain_loss_weight * pain_loss
            added_loss = True
        if valid_stim_mask.any():
            total_loss = total_loss + self.hparams.stim_loss_weight * stim_loss if added_loss else self.hparams.stim_loss_weight * stim_loss
            added_loss = True
        if not added_loss:
            # No valid labels in batch, grad hack as before
            total_loss = (pain_logits.sum() + stimulus_logits.sum()) * 0.0

        self.log(f"{stage}_loss", total_loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        loss = self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='train', batch_idx=batch_idx)
        # Log learning rate
        self.log("learning_rate", self.hparams.learning_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='val', batch_idx=batch_idx)
        # No loss returned for validation/test

    def test_step(self, batch, batch_idx):
        features, pain_labels, stimulus_labels = batch
        pain_logits, stimulus_logits = self(features)
        self._calculate_loss_and_metrics(pain_logits, stimulus_logits, pain_labels, stimulus_labels, stage='test', batch_idx=batch_idx)
        # No loss returned for validation/test

    def sanity_check(self, batch_size: int = 4):
        """Performs a basic check on input/output shapes."""
        print(f"Running sanity check for {self.__class__.__name__}...")
        try:
            # Ensure model is on the correct device for the check
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.to(device)
            # Using self.device which Lightning manages
            device = self.device
            dummy_input = torch.randn(batch_size, self.hparams.input_dim, device=device)
            self.eval() # Set model to evaluation mode for the check
            with torch.no_grad(): # No need to track gradients
                pain_logits, stimulus_logits = self(dummy_input)
            self.train() # Set back to train mode

            # Check output shapes
            expected_pain_shape = (batch_size, self.hparams.num_pain_classes - 1)
            expected_stim_shape = (batch_size, self.hparams.num_stimulus_classes - 1)

            assert pain_logits.shape == expected_pain_shape, \
                f"Pain logits shape mismatch: Expected {expected_pain_shape}, Got {pain_logits.shape}"
            assert stimulus_logits.shape == expected_stim_shape, \
                f"Stimulus logits shape mismatch: Expected {expected_stim_shape}, Got {stimulus_logits.shape}"

            print(f"Sanity check passed for {self.__class__.__name__}.")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Pain logits shape: {pain_logits.shape}")
            print(f"  Stimulus logits shape: {stimulus_logits.shape}")
            # Print head weight norms for debug/diagnostics
            print(f"  Pain head weights: {self.pain_head.weight.data.norm():.4f}")
            print(f"  Stim head weights: {self.stimulus_head.weight.data.norm():.4f}")

        except Exception as e:
            print(f"Sanity check failed for {self.__class__.__name__}: {e}")
            raise # Re-raise the exception after printing

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use."""
        # Explicitly cast learning rate to float
        try:
            lr = float(self.hparams.learning_rate)
            wd = float(self.hparams.weight_decay)
        except ValueError:
            print(f"Error: Could not convert learning rate '{self.hparams.learning_rate}' to float!")
            raise # Re-raise the error
            
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        
        # Check if lr_scheduler parameters are available in hparams
        if hasattr(self.hparams, 'use_lr_scheduler') and self.hparams.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max' if hasattr(self, 'monitor_metric') and 'QWK' in self.monitor_metric else 'min',
                factor=getattr(self.hparams, 'lr_factor', 0.5),
                patience=getattr(self.hparams, 'lr_patience', 5),
                min_lr=getattr(self.hparams, 'min_lr', 1e-6)
            )
            
            # PyTorch Lightning expects this format for ReduceLROnPlateau
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": getattr(self, 'monitor_metric', 'val_pain_QWK'),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                }
            }
        
        return optimizer

# Remove the old if __name__ == '__main__': block if it exists
# (The edit tool should replace the entire file content based on the instruction) 