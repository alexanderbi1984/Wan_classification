# Multi-Task Pipeline: Issues and Improvements

This document outlines potential issues and suggested improvements for the multi-task training pipeline for pain/stimulus classification.

## Current Issues

### 1. Input Dimension Detection

The current approach to detecting input dimensions has several potential issues:

```python
input_shape = syracuse_dm.example_shape
if input_shape is None and syracuse_dm.train_dataset and len(syracuse_dm.train_dataset) > 0:
    input_shape = syracuse_dm.train_dataset.example_shape

if input_shape is None:
    if biovid_dm.example_shape is not None:
        input_shape = biovid_dm.example_shape
    elif biovid_dm.train_dataset and len(biovid_dm.train_dataset) > 0:
        input_shape = biovid_dm.train_dataset.example_shape
    else:
        raise ValueError("Cannot determine input_shape from either Syracuse or BioVid datasets.")
```

- **Fragile error handling**: Multiple nested conditionals make this section hard to follow
- **Inconsistent shape handling**: Shape detection doesn't consistently handle the case when temporal_pooling="none" and flatten=True
- **No early validation**: Shape assumptions aren't validated early, potentially causing cryptic errors later in training

### 2. Empty Validation Dataset Handling

```python
if syracuse_dm.val_dataset and len(syracuse_dm.val_dataset) > 0:
    wrapped_syracuse_val = CombinedTaskDatasetWrapper(syracuse_dm.val_dataset, task_name='pain_level')
    val_ds_for_loader = wrapped_syracuse_val
else:
    print("[WARN] Syracuse validation dataset is empty. Using dummy empty list for val_loader.")
    val_ds_for_loader = []
```

- The empty list approach can lead to subtle bugs; an empty dataset with proper structure would be more robust
- Early stopping based on validation metrics won't work properly without validation data

### 3. Configuration Structure

- The config structure has inconsistencies between parameter naming in YAML vs code:
  - `optimizer_type` in YAML vs `optimizer_name` in model
  - `model_params.num_pain_classes` should be linked to `syracuse_settings.num_classes_pain`

### 4. Hyperparameter Management

- Dynamic class weights are handled inconsistently
- No clear method for hyperparameter search integration (e.g., with Ray Tune)

### 5. Error Handling 

- Limited error handling for corrupt data files or missing features
- No graceful recovery if a single batch fails during training

## Suggested Improvements

### 1. Robust Input Dimension Detection

```python
def detect_input_shape(primary_dm, fallback_dm=None):
    """Robustly detect input shape from data modules with clear error messages."""
    for dm in [primary_dm, fallback_dm]:
        if dm is None:
            continue
            
        # Try direct example_shape first
        if dm.example_shape is not None:
            return dm.example_shape
            
        # Fall back to dataset example shape
        if dm.train_dataset and hasattr(dm.train_dataset, 'example_shape'):
            if dm.train_dataset.example_shape is not None:
                return dm.train_dataset.example_shape
                
    # If we get here, no valid shape found
    raise ValueError(f"Could not detect input shape from data modules: {primary_dm}, {fallback_dm}")
```

### 2. Better Validation Set Handling

```python
def create_validation_set(dataset, min_validation_samples=10):
    """Create a valid validation set, even if the original is empty."""
    if dataset is None or len(dataset) == 0:
        # Log warning and create a small dummy validation set from training
        warnings.warn("No validation data available, creating a minimal validation set from training data.")
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > min_validation_samples:
            # Use a small portion of training data for validation when needed
            indices = list(range(min(min_validation_samples, len(self.train_dataset))))
            return torch.utils.data.Subset(self.train_dataset, indices)
    return dataset
```

### 3. Configuration and Hyperparameter Management

- Use direct parameter binding with schema validation:

```python
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    input_dim: Optional[int] = None
    num_pain_classes: int
    num_stimulus_classes: int
    encoder_hidden_dims: List[int] = [512, 256]
    # ...
    
    @validator('num_pain_classes')
    def validate_num_pain_classes(cls, v, values):
        if v <= 1:
            raise ValueError("num_pain_classes must be at least 2")
        return v
```

### 4. Enhanced Debugging and Logging

- Add debugging capabilities for model internals:

```python
def get_internal_activations(self, x, pain_labels=None, stimulus_labels=None):
    """Return internal activations for debugging."""
    encoded = self.shared_encoder(x)
    pain_logits = self.pain_head(encoded)
    stimulus_logits = self.stimulus_head(encoded)
    
    return {
        "encoded_features": encoded,
        "pain_logits": pain_logits,
        "stimulus_logits": stimulus_logits,
        "pain_probs": torch.sigmoid(pain_logits) if pain_logits is not None else None,
        "stim_probs": torch.sigmoid(stimulus_logits) if stimulus_logits is not None else None,
    }
```

### 5. Improved Error Handling and Resilience

- Add dataset validation step:

```python
def validate_dataset(dataset, name="dataset"):
    """Validate a dataset by attempting to fetch a few samples."""
    if dataset is None or len(dataset) == 0:
        warnings.warn(f"{name} is empty")
        return False
        
    # Try to fetch the first few samples
    try:
        for i in range(min(3, len(dataset))):
            _ = dataset[i]
        return True
    except Exception as e:
        warnings.error(f"Error validating {name}: {e}")
        return False
```

### 6. Performance Optimization

- Implement feature caching to speed up repeated access:

```python
class CachedDataset(torch.utils.data.Dataset):
    """Wrapper that caches samples in memory for faster access."""
    def __init__(self, dataset, cache_size=1000):
        self.dataset = dataset
        self.cache_size = min(cache_size, len(dataset))
        self.cache = {}
        
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        item = self.dataset[idx]
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
            
        return item
        
    def __len__(self):
        return len(self.dataset)
```

### 7. Enhanced Training Features

- Add support for mixed precision training
- Implement gradient accumulation for larger effective batch sizes
- Add learning rate warmup period

### 8. Documentation and Monitoring

- Add model parameter count and FLOPS calculation
- Add throughput benchmarking to optimize batch size
- Generate model architecture visualization

### 9. Testing and Validation

- Create unit tests for critical components
- Add data consistency checks
- Implement a sanity check for early iterations (validate loss decreases)

## Implementation Priority

1. **High Priority**
   - Input dimension detection robustness
   - Configuration structure consistency
   - Validation dataset handling

2. **Medium Priority**
   - Error handling improvements
   - Performance optimizations
   - Enhanced debugging

3. **Low Priority**
   - Advanced training features
   - Documentation and monitoring tools
   - Comprehensive testing framework 