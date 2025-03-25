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



class ReplaceLRU():
	"""
	Adapted from https://github.com/minyoungg/vqtorch/blob/main/vqtorch/nn/utils/replace.py
	Attributes:
		rho (float): mutation noise
		timeout (int): number of batch it has seen
	"""
	VALID_POLICIES = ['input_random', 'input_kmeans', 'self']

	def __init__(self, rho=1e-4, timeout=100):
		assert timeout > 1
		assert rho > 0.0
		self.rho = rho
		self.timeout = timeout

		self.policy = 'input_random'
		# self.policy = 'input_kmeans'
		# self.policy = 'self'
		self.tau = 2.0

		assert self.policy in self.VALID_POLICIES
		return

	@staticmethod
	def apply(module, rho=0., timeout=100):
		""" register forward hook """
		fn = ReplaceLRU(rho, timeout)
		device = next(module.parameters()).device
		module.register_forward_hook(fn)
		module.register_buffer('_counts', timeout * torch.ones(module.num_codewords))
		module._counts = module._counts.to(device)
		return fn

	def __call__(self, module, inputs, outputs):
		"""
		This function is triggered during forward pass
		
		Args
			module (VectorQuantize)
			inputs (tuple): A tuple with 1 element
				z_e (Tensor)
			outputs:
				vq_out (dict)
		"""
		if not module.training:
			return
		
		if hasattr(module, 'is_freezed'):
			if module.is_freezed.item() == 1: return

		# count down all code by 1 and if used, reset timer to timeout value
		module._counts -= 1

		# --- computes most recent codebook usage --- #
		unique, counts = torch.unique(outputs['q'], return_counts=True)
		module._counts.index_fill_(0, unique, self.timeout)

		# --- find how many needs to be replaced --- #
		# num_active = self.check_and_replace_dead_codes(module, outputs)
		inactive_indices = torch.argwhere(module._counts == 0).squeeze(-1)
		num_inactive = inactive_indices.size(0)

		if num_inactive > 0:

			if self.policy == 'self':
				# exponential distance allows more recently used codes to be even more preferable
				p = torch.zeros_like(module._counts)
				p[unique] = counts.float()
				p = p / p.sum()
				p = torch.exp(self.tau * p) - 1 # the negative 1 is to drive p=0 to stay 0

				selected_indices = torch.multinomial(p, num_inactive, replacement=True)
				selected_values = module.codebook.data[selected_indices].clone()

			elif self.policy == 'input_random':
				z_e = inputs[0].flatten(0, -2)   # flatten to 2D
				z_e = z_e[torch.randperm(z_e.size(0))] # shuffle
				mult = num_inactive // z_e.size(0) + 1
				if mult > 1: # if theres not enough
					z_e = torch.cat(mult * [z_e])
				selected_values = z_e[:num_inactive]

			elif self.policy == 'input_kmeans':
				# can be extremely slow
				from torchpq.clustering import KMeans
				z_e = inputs[0].flatten(0, -2)   # flatten to 2D
				z_e = z_e[torch.randperm(z_e.size(0))] # shuffle
				kmeans = KMeans(n_clusters=num_inactive, distance='euclidean', init_mode="kmeans++")
				kmeans.fit(z_e.data.T.contiguous())
				selected_values = kmeans.centroids.T

			if self.rho > 0:
				norm = selected_values.norm(p=2, dim=-1, keepdim=True)
				noise = torch.randn_like(selected_values)
				selected_values = selected_values + self.rho * norm * noise

			# --- update dead codes with new codes --- #
			module.codebook.data[inactive_indices] = selected_values
			module._counts[inactive_indices] += self.timeout

		return outputs
	

def lru_replacement(vq_module, rho=1e-4, timeout=100):
	
	ReplaceLRU.apply(vq_module, rho, timeout)
	return vq_module