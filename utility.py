import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvexAggregation(nn.Module):
    def __init__(self, input_dim=512):
        super(ConvexAggregation, self).__init__()
        self.weight_params = nn.Parameter(torch.randn(input_dim))  # Learnable weights

    def forward(self, patch_features):
        """
        patch_features: Tensor of shape (M, D) where M is selected patches and D is feature dimension
        """
        w_i = F.softmax(self.weight_params, dim=0)  # Ensure non-negative and sum-to-one
        H_WSI = torch.sum(w_i.unsqueeze(1) * patch_features, dim=0)  # Weighted sum

        return H_WSI  # Global WSI-level representation


class TotalLossFunction(nn.Module):
    def __init__(self, alpha=0.1):
        super(TotalLossFunction, self).__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds, targets, laplacian_matrix, feature_embeddings):
        ce_loss = self.cross_entropy(preds, targets)

        # Laplacian Regularization Loss
        laplacian_loss = torch.sum(laplacian_matrix * (feature_embeddings.unsqueeze(1) - feature_embeddings.unsqueeze(0)).pow(2))

        total_loss = ce_loss + self.alpha * laplacian_loss
        return total_loss

from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, precision_score, recall_score

def evaluate_model(model, dataloader):
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"AUC: {auc:.3f}, Kappa: {kappa:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    return auc, kappa, f1, precision, recall


def train_model(irm, glat, conv_agg, train_loader, val_loader, optimizer, loss_fn, num_epochs=100):
    """
    Trains the model using IRM, GLAT, and Convex Aggregation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    irm, glat, conv_agg = irm.to(device), glat.to(device), conv_agg.to(device)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        irm.train()
        glat.train()
        conv_agg.train()

        total_loss = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)

            # Patch selection using IRM
            selected_patches = irm(patches)
            
            # Feature refinement using GLAT
            refined_features = glat(selected_patches)

            # Global representation using Convex Aggregation
            WSI_representation = conv_agg(refined_features)

            # Classification
            outputs = nn.Linear(WSI_representation.shape[1], len(set(labels.tolist())))(WSI_representation)

            # Compute Loss
            loss = loss_fn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = validate_model(irm, glat, conv_agg, val_loader, loss_fn)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(irm, glat, conv_agg, optimizer, "best_model.pth")

    print("Training completed!")

def validate_model(irm, glat, conv_agg, val_loader, loss_fn):
    """
    Evaluates the model on validation data.
    """
    irm.eval()
    glat.eval()
    conv_agg.eval()

    total_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(device), labels.to(device)

            # Patch selection using IRM
            selected_patches = irm(patches)

            # Feature refinement using GLAT
            refined_features = glat(selected_patches)

            # Global representation using Convex Aggregation
            WSI_representation = conv_agg(refined_features)

            # Classification
            outputs = nn.Linear(WSI_representation.shape[1], len(set(labels.tolist())))(WSI_representation)

            # Compute Loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def save_checkpoint(irm, glat, conv_agg, optimizer, filename="model_glat.pth"):
    """
    Saves the model's current state.
    """
    checkpoint = {
        "irm_state_dict": irm.state_dict(),
        "glat_state_dict": glat.state_dict(),
        "conv_agg_state_dict": conv_agg.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(irm, glat, conv_agg, optimizer, filename="model_glat.pth"):
    """
    Loads a saved model checkpoint.
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        irm.load_state_dict(checkpoint["irm_state_dict"])
        glat.load_state_dict(checkpoint["glat_state_dict"])
        conv_agg.load_state_dict(checkpoint["conv_agg_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded: {filename}")
    else:
        print(f"No checkpoint found at {filename}")

def plot_training_curves(train_losses, val_losses):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.show()

def visualize_attention_map(attention_matrix, title="Attention Map"):
    """
    Displays an attention heatmap.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Patches")
    plt.ylabel("Patches")
    plt.show()
