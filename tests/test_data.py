from torch.utils.data import Dataset

from mnist_classifier.data import normalize


import torch

def test_normalize():
    # Create a sample tensor
    images = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalized_images = normalize(images)
    
    # Assert mean close to 0 and std close to 1
    assert torch.isclose(normalized_images.mean(), torch.tensor(0.0), atol=1e-4), "Mean should be close to 0"
    assert torch.isclose(normalized_images.std(), torch.tensor(1.0), atol=1e-4), "Std should be close to 1"
