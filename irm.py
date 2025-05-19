import torch.nn as nn
import torch.nn.functional as F
import timm


class IterativeRefinementModule(nn.Module):
    """
    Iterative Refinement Module (IRM) for progressive patch selection.
    Uses a pretrained ResNet50 and a frozen Foundation Model (FM).
    """

    def __init__(self, resnet50, foundation_model, num_iterations=5, num_selected_patches=50):
            super(IterativeRefinementModule, self).__init__()
            self.resnet50 = resnet50
            self.foundation_model = foundation_model  # UNI Model (frozen, no gradients)
            self.num_iterations = num_iterations  # Number of refinement iterations
            self.num_selected_patches = num_selected_patches  # Number of patches selected per iteration


    def forward(self, patches):
        """
        Iteratively refines patch selection.
        Args:
            patches (Tensor): Input patches of shape (B, N, C, H, W)
                              B = Batch size, N = Total patches per WSI
        Returns:
            Tensor: Refined selected patches (B, num_selected_patches, C, H, W)
        """

        batch_size, num_patches, C, H, W = patches.shape

        # Step 1: Extract features using ResNet50
        patches = patches.view(-1, C, H, W)  # Flatten batch & patch dims
        features = self.resnet50(patches)  # Shape: (B*N, D)
        features = features.view(batch_size, num_patches, -1)  # Reshape back

        # Step 2: Initialize importance scores
        importance_scores = torch.zeros(batch_size, num_patches).to(patches.device)

        for t in range(self.num_iterations):
            with torch.no_grad():  # FM operates in no-gradient mode
                attention_weights = self.foundation_model(features)  # Compute contextual importance scores

            # Step 3: Normalize scores using softmax
            importance_scores = F.softmax(attention_weights, dim=1)  # Shape: (B, N)

            # Step 4: Select top M patches based on importance scores
            _, top_patch_indices = torch.topk(importance_scores, self.num_selected_patches, dim=1)

            # Step 5: Gather selected patches and their features
            selected_patches = torch.gather(patches.view(batch_size, num_patches, C, H, W), 1, 
                                            top_patch_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, H, W))

            # Step 6: Update features using selected patches
            features = torch.gather(features, 1, top_patch_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))

        return selected_patches


