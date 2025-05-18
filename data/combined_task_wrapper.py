from torch.utils.data import Dataset
import torch

class CombinedTaskDatasetWrapper(Dataset):
    def __init__(self, original_dataset, task_name: str, multimodal: bool = False):
        """Wraps an original dataset (Syracuse or BioVid) to output a standardized
        tuple for the multi-task model.
        If multimodal=True, expects the original dataset to return (vae_x, xdit_x, ...),
        and outputs (vae_x, xdit_x, pain_label, stimulus_label).
        If multimodal=False, outputs (features, pain_label, stimulus_label) for backward compatibility.
        Labels for the non-relevant task are set to -1 (integer).
        Args:
            original_dataset: The dataset instance (SyracuseDataset or BioVidDataset).
            task_name (str): The name of the primary task for this dataset, either 'pain_level' (for Syracuse) 
                         or 'stimulus' (for BioVid).
            multimodal (bool): If True, expects the dataset to return (vae_x, xdit_x, ...).
        """
        self.original_dataset = original_dataset
        if task_name not in ['pain_level', 'stimulus']:
            raise ValueError("task_name must be 'pain_level' or 'stimulus'")
        self.task_name = task_name
        self.multimodal = multimodal

        # Warning if Syracuse dataset (task_name='pain_level') is not in classification mode
        if self.task_name == 'pain_level':
            if hasattr(self.original_dataset, 'task') and self.original_dataset.task != 'classification':
                print(f"[WARN] CombinedTaskDatasetWrapper expects SyracuseDataset (task_name='pain_level') "
                      f"to be in 'classification' mode for integer pain labels. "
                      f"Current mode: {self.original_dataset.task}. Ensure pain_label output is an integer class index.")
            if hasattr(self.original_dataset, 'task') and self.original_dataset.task == 'classification' and \
               (not hasattr(self.original_dataset, 'thresholds') or self.original_dataset.thresholds is None):
                print(f"[WARN] SyracuseDataset (task_name='pain_level') is in 'classification' mode "
                      f"but thresholds might be missing. Ensure class labels are correctly generated.")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_item = self.original_dataset[idx]
        if self.multimodal:
            # Expect (vae_x, xdit_x, ...)
            vae_x = original_item[0]
            xdit_x = original_item[1]
            pain_target = -1
            stimulus_target = -1
            if self.task_name == 'pain_level':  # Syracuse
                pain_label_data = original_item[2]
                if isinstance(pain_label_data, torch.Tensor):
                    pain_target = pain_label_data.item()
                elif isinstance(pain_label_data, (int, float)):
                    pain_target = int(pain_label_data)
                else:
                    raise TypeError(f"Syracuse (pain_level) pain_class_label is of unexpected type: {type(pain_label_data)}.")
            elif self.task_name == 'stimulus':  # BioVid
                stimulus_label_data = original_item[3]
                if isinstance(stimulus_label_data, torch.Tensor):
                    stimulus_target = stimulus_label_data.item()
                elif isinstance(stimulus_label_data, (int, float)):
                    stimulus_target = int(stimulus_label_data)
                else:
                    raise TypeError(f"BioVid (stimulus) stimulus_label is of unexpected type: {type(stimulus_label_data)}.")
            return vae_x, xdit_x, pain_target, stimulus_target
        else:
            # Old behavior: single feature tensor
            features = None
            pain_target = -1
            stimulus_target = -1
            if self.task_name == 'pain_level':  # Syracuse
                features = original_item[0]
                pain_label_data = original_item[2]
                if isinstance(pain_label_data, torch.Tensor):
                    pain_target = pain_label_data.item()
                elif isinstance(pain_label_data, (int, float)):
                    pain_target = int(pain_label_data)
                else:
                    raise TypeError(f"Syracuse (pain_level) pain_class_label is of unexpected type: {type(pain_label_data)}.")
            elif self.task_name == 'stimulus':  # BioVid
                features = original_item[0]
                stimulus_label_data = original_item[2]
                if isinstance(stimulus_label_data, torch.Tensor):
                    stimulus_target = stimulus_label_data.item()
                elif isinstance(stimulus_label_data, (int, float)):
                    stimulus_target = int(stimulus_label_data)
                else:
                    raise TypeError(f"BioVid (stimulus) stimulus_label is of unexpected type: {type(stimulus_label_data)}.")
            return features, pain_target, stimulus_target

    @property
    def example_shape(self):
        if hasattr(self.original_dataset, 'example_shape') and self.original_dataset.example_shape is not None:
            return self.original_dataset.example_shape
        if len(self.original_dataset) > 0:
            try:
                original_item = self.original_dataset[0]
                # For multimodal, return shape of vae_x and xdit_x
                if self.multimodal:
                    return (original_item[0].shape, original_item[1].shape)
                else:
                    return original_item[0].shape
            except Exception as e:
                print(f"[WARN] Could not infer example_shape from CombinedTaskDatasetWrapper for {self.task_name}: {e}")
                return None
        return None

    def __repr__(self):
        return f"<CombinedTaskDatasetWrapper task_name='{self.task_name}' original_dataset={repr(self.original_dataset)}>" 