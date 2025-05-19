import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLaplacianAttentionTransformer(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_heads=8, sigma=0.5, lambda_lap=0.1):
        super(GraphLaplacianAttentionTransformer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.lambda_lap = lambda_lap

        # Self-attention layers
        self.W_Q = nn.Linear(input_dim, hidden_dim)
        self.W_K = nn.Linear(input_dim, hidden_dim)
        self.W_V = nn.Linear(input_dim, hidden_dim)

        # Learnable Laplacian Filter
        self.L_theta = nn.Parameter(torch.eye(hidden_dim))

        # Final Linear Projection
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, patch_features):
        """
        patch_features: Tensor of shape (N, D)
        """
        N, D = patch_features.shape

        # Compute adjacency matrix W using Gaussian kernel
        pairwise_distances = torch.cdist(patch_features, patch_features, p=2)
        W = torch.exp(-pairwise_distances ** 2 / (2 * self.sigma ** 2))

        # Compute Graph Laplacian
        D_matrix = torch.diag(W.sum(dim=1))
        L_global = D_matrix - W

        # Compute Self-Attention
        Q = self.W_Q(patch_features)
        K = self.W_K(patch_features)
        V = self.W_V(patch_features)

        # Apply Learnable Laplacian Filtering
        Q_prime = torch.matmul(self.L_theta, Q.T).T
        K_prime = torch.matmul(self.L_theta, K.T).T
        V_prime = torch.matmul(self.L_theta, V.T).T

        # Graph Laplacian Attention (GLA)
        attention_logits = torch.matmul(Q_prime, K_prime.T) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attention_logits += self.lambda_lap * L_global
        A_prime = F.softmax(attention_logits, dim=-1)

        H = torch.matmul(A_prime, V_prime)
        H = self.fc_out(H)

        return H
