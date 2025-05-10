from classifier.multi_task_coral import MultiTaskCoralClassifier
import torch.optim as optim

class BioVidCoralClassifier(MultiTaskCoralClassifier):
    """
    A version of MultiTaskCoralClassifier specifically for BioVid dataset.
    Ensures LR scheduler uses stimulus metrics instead of pain metrics.
    """
    def configure_optimizers(self):
        """Configure optimizer and LR scheduler with stimulus metrics."""
        try:
            lr = float(self.hparams.learning_rate)
            wd = float(self.hparams.weight_decay)
        except ValueError:
            print(f"Error: Could not convert learning rate '{self.hparams.learning_rate}' to float!")
            raise
            
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")
        
        # Check if lr_scheduler parameters are available in hparams
        if hasattr(self.hparams, 'use_lr_scheduler') and self.hparams.use_lr_scheduler:
            # Always use stimulus metrics for monitoring
            monitor_metric = "val_stim_QWK"
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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