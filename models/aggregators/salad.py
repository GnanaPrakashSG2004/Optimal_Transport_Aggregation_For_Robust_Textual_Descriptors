import torch
import torch.nn as nn

# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m*one).to(scores), (n*one).to(scores), (torch.abs((n-m)*one)).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    
    couplings = torch.cat([scores, bins], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), (bs).log()[None] + norm])

    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class SinkhornAggregator(nn.Module):
    """
    Class for the Sinkhorn Aggregator module.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int):  The number of channels of the clusters (l).
        token_dim (int):    The dimension of the CLS token (c).
        dropout (float):    The dropout rate.
    """
    def __init__(self,
            num_channels=768,
            num_clusters=32,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim  = cluster_dim
        self.token_dim    = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for CLS token t_{n+1}
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )

        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Linear(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Linear(512, self.cluster_dim, 1)
        )

        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Linear(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Linear(512, self.num_clusters, 1),
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x):
        """
        Forward pass of the Sinkhorn Aggregator module to compute the global descriptor.

        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, n, hidden_size].
            (torch.Tensor): The token tensor (t_{n+1}) [B, hidden_size].

        where n is the number of encoder hidden states of the backbone.

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + c].
        """
        x, t = x # Extract features and token

        f = self.cluster_features(x).permute(0, 2, 1) # [B, l, n]
        p = self.score(x).permute(0, 2, 1) # [B, m, n]
        t = self.token_features(t)   # [B, c]

        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3) # [B, m + 1, n] -> +1 is from the dustbin cluster
        p = torch.exp(p)
        p = p[:, :-1, :] # Remove dustbin cluster

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # [B, m, n] -> [B, l, m, n]
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1) # [B, l, n] -> [B, l, m, n]

        f = torch.cat([
            # Next two lines are L2 intra-normalization (normalization within the sample)

            nn.functional.normalize(t, p=2, dim=-1),

            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
            # Sum of all local features extracted for each hidden state of the sequence weighted by the cluster probabilities
            # Then normalize the features within each sample and flatten them so that there are only B flattened vectors - [B, m*l]

        ], dim=-1) # [B, c + m*l] -> Concatenate the global token and the aggregated flattened local features

        return nn.functional.normalize(f, p=2, dim=-1)  # Inter-L2 normalization (across flattened local + global descriptor) gives the final global descriptor for each sample in the batch

if __name__ == "__main__":
    # Test the Sinkhorn Aggregator
    model = SinkhornAggregator()
    x = (torch.rand(2, 10, 768), torch.rand(2, 768))
    print(model(x).shape) # Should print [2, 384]