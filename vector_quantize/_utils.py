import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dist(z_e, codebook, cos_dist=False, z_proj_matrix=None, c_proj_matrix=None):
    """
    Args:
        z_e (Tensor): flattened latent with shape (bsz, *, z_dim), * denotes flattened quantized dimensions
        codebook (Tensor): embedding weight tensor with shape (num_codewords, c_dim)
        cos_dist (bool): whether to use cosine distance or not
        z_proj_matrix (Tensor): latent projection matrix with shape (z_dim, proj_dim) for low-dimensional search
        c_proj_matrix (Tensor): codebook projection matrix with shape (c_dim, proj_dim) for low-dimensional search

    Returns:
        dists (Tensor): distance between z_e and each codewords in codebook with shape (bsz, *, num_codewords)
    """

    if cos_dist:
        z_e = F.normalize(z_e, p=2, dim=-1)
        codebook = F.normalize(codebook, p=2, dim=-1)

    if z_proj_matrix is not None:
        z_e = z_e @ z_proj_matrix

    if c_proj_matrix is not None:
        codebook = codebook @ c_proj_matrix

    dists = torch.cdist(z_e, codebook, p=2) # (bsz, *, num_codewords)
    return dists


def freeze_dict_forward_hook(module, inputs, outputs):
    if not module.training or module.is_freezed.item() == 0:
        return
    
    z_e = inputs[0]
    outputs = {
        'z_q': z_e,
        'q': None,
        'cm_loss': torch.zeros(z_e.shape[0], device=z_e.device),
        'cb_loss': torch.zeros(z_e.shape[0], device=z_e.device),
    }
    return outputs


def init_proj_matrix(random_proj, dim, proj_dim):
    if random_proj:
        proj_matrix = torch.empty(dim, proj_dim)
        nn.init.xavier_normal_(proj_matrix)
    else:
        proj_matrix = nn.Parameter(torch.empty(dim, proj_dim), requires_grad=True)
        nn.init.kaiming_normal_(proj_matrix)
    return proj_matrix