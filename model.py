from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from utils import gather
from unet import UNet


class DenoiseDiffusion(nn.Module):
    """
    Denoise Diffusion Model
    """
    def __init__(self, params: dict):
        super(DenoiseDiffusion, self).__init__()

        self.T =  params['T'] #
        self.device = params['device']
        self.eps_model = UNet(image_channels=params['image_channels'],
                              n_channels=params['n_channels'],
                              ch_mults=params['channel_multipliers'],
                              is_attn=params['is_attention']).to(self.device)

        # create $\beta_1, \dots, \beta_T$
        self.beta = torch.linspace(0.0001, 0.02, self.T).to(self.device)

        self.alpha = 1.0 - self.beta

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]=None):
        """
        Sample from  q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1 - \bar\alpha_t)I)
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean + eps * (var ** 0.5)

    def p_sample(self, xt: torch.Tensor, t:torch.Tensor):
        """
        Sample from p(x_{t-1}|x_t)
        """
        eps_theta = self.eps_model(xt, t)
        alpha_coef = 1. / (gather(self.alpha, t) ** 0.5)
        eps_coef =  gather(self.beta, t) / (1 - gather(self.alpha_bar, t)) ** 0.5
        mean = alpha_coef * (xt - eps_coef * eps_theta)

        # Question: how to get var? [Answer]: var is sigma^2, which is self.beta, why?
        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)

        return mean + eps * (var ** 0.5)


    def loss(self, x0: torch.Tensor, eps: Optional[torch.Tensor]=None):
        """
        Loss calculation
        x0: (B, ...)
        """
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device, dtype=torch.long)
        # Note that in the paper, t \in [1, T], but in the code, t \in [0, T-1]
        if eps is None: eps = torch.randn_like(x0)
        xt = self.q_xt_x0(x0, t, eps)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(eps, eps_theta)
















