import torch
from torch import nn


class WNormLoss(nn.Module):

	def __init__(self, opts):
		super(WNormLoss, self).__init__()
		self.opts = opts

	def forward(self, latent, latent_avg=None):
		if self.opts.start_from_latent_avg or self.opts.start_from_encoded_w_plus:
			latent = latent - latent_avg
		return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
