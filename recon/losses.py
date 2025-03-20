import torch
import torch.nn.functional as F


def d_loss(logits_real, logits_fake, method='vanilla'):
    if method == 'vanilla':
        d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    elif method == 'hinge':
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return d_loss


def g_loss(logits_fake):
    g_loss = -torch.mean(logits_fake)
    return g_loss