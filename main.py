import torch
import torch.nn as nn
import torch.optim as optim
from irm import IterativeRefinementModule
from glat import GraphLaplacianAttentionTransformer
from utility import ConvexAggregation, TotalLossFunction, train_model, validate_model
from dataloader import get_dataloader

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
NUM_FOLDS = 5  # Number of folds for cross-validation

def main():
    """
    Main training script for prostate cancer grading model using five-fold cross-validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cross-validation data
    tcga_folds, sicap_folds = get_dataloader(batch_size=BATCH_SIZE, num_folds=NUM_FOLDS)

    # Correct: Use the dataloaders returned from get_dataloader()
    for fold, ((train_loader_tcga, val_loader_tcga), (train_loader_sicap, val_loader_sicap)) in enumerate(zip(tcga_folds, sicap_folds)):
        print(f"\n=== Training Fold {fold+1}/{NUM_FOLDS} ===")

        # Combine TCGA and SICAP data loaders directly
        train_loaders = [train_loader_tcga, train_loader_sicap]
        val_loaders = [val_loader_tcga, val_loader_sicap]

        # Train and validate using these loaders
        train_loss = sum(train_model(irm, glat, conv_agg, loader, optimizer, loss_fn) for loader in train_loaders) / len(train_loaders)
        val_loss = sum(validate_model(irm, glat, conv_agg, loader, loss_fn) for loader in val_loaders) / len(val_loaders)

        print(f"Fold {fold+1} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


        # Load pretrained models
        resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        foundation_model = torch.hub.load("mahmoodlab/UNI", "uni_model").eval()  # Frozen model

        # Define modules
        irm = IterativeRefinementModule(resnet50, foundation_model).to(device)
        glat = GraphLaplacianAttentionTransformer().to(device)
        conv_agg = ConvexAggregation().to(device)

        # Define loss function & optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(irm.parameters()) + list(glat.parameters()) + list(conv_agg.parameters()), 
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Early stopping setup
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(NUM_EPOCHS):
            train_loss = 0
            val_loss = 0

            for train_loader in train_loaders:
                train_loss += train_model(irm, glat, conv_agg, train_loader, optimizer, loss_fn)

            for val_loader in val_loaders:
                val_loss += validate_model(irm, glat, conv_agg, val_loader, loss_fn)

            train_loss /= len(train_loaders)
            val_loss /= len(val_loaders)

            print(f"Fold {fold+1} | Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model per fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter
                torch.save({
                    "irm": irm.state_dict(),
                    "glat": glat.state_dict(),
                    "conv_agg": conv_agg.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, f"best_model_fold_{fold+1}.pth")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered for Fold {fold+1}. Stopping training.")
                break

if __name__ == "__main__":
    main()



