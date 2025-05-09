# Classifier Folder

This folder contains modules and configs for PyTorch Lightning-based ordinal and multi-task classifiers.

## Contents
- **multi_task_coral.py**: Main implementation of `MultiTaskCoralClassifier`, a flexible multi-task model supporting CORAL for ordinal regression in pain/stimulus multi-task settings. Includes robust, configurable architectures, metric logging, and advanced loss options.
- **config_multi_task_coral.yaml**: Structured sample YAML for controlling all model, optimizer, and training settings. Add new configs here to manage experiment variants (e.g., with/without LoRA, different heads, etc).

## Usage

1. **Edit/Create a YAML config:**
    - Adjust values in `config_multi_task_coral.yaml` (or create your own, e.g. for LoRA versions):
    ```yaml
    model:
      input_dim: 768
      num_pain_classes: 5
      num_stimulus_classes: 5
      encoder_hidden_dims: [512, 256]
      encoder_dropout: 0.5
    optimizer:
      optimizer_name: AdamW
      learning_rate: 0.0001
      weight_decay: 0.0
    # ...
    ```

2. **Load config in Python and instantiate the model:**
    ```python
    import yaml
    from classifier.multi_task_coral import MultiTaskCoralClassifier
    with open('classifier/config_multi_task_coral.yaml') as f:
        config = yaml.safe_load(f)
    hparams = {**config['model'], **config['optimizer'], **config['loss']}
    model = MultiTaskCoralClassifier(**hparams)
    ```

3. **Extending (e.g., LoRA):**
    - Add new model arguments/config block (e.g. `use_lora`, `lora_r`, ...) to a new YAML file, and update your model to use those when enabled.

4. **Tips:**
    - Keep architecture-specific arguments clearly grouped and named in your configs (e.g., `encoder_dropout` for only the encoder).
    - Copy and adapt the provided `config_multi_task_coral.yaml` to create variants for new model extensions or ablation studies.

## Key Dependencies
- torch, pytorch-lightning, torchmetrics, PyYAML

See module and config docstrings for further usage notes and details. 