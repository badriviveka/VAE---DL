import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    """
    TODO: Implement L2 normalization.
    """
    normalized = v / (v.norm() + eps)
    return normalized


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        """
        Reference:
        SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1802.05957.pdf
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        """
        TODO: Implement Spectral Normalization
        Hint: 1: Use getattr to first extract u, v, w.
              2: Apply power iteration.
              3: Calculate w with the spectral norm.
              4: Use setattr to update w in the module.
        """
        u = getattr(self.module, self.name + "_uval")
        v = getattr(self.module, self.name + "_vval")
        w = getattr(self.module, self.name + "_wval")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            wT = w.view(height,-1).data
            v.data = l2normalize(torch.mv(torch.t(wT), u.data))
            u.data = l2normalize(torch.mv(wT, v.data))
        sigma_val = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma_val.expand_as(w))



    def _make_params(self):
        """
        No need to change. Initialize parameters.
        v: Initialize v with a random vector (sampled from isotropic distrition).
        u: Initialize u with a random vector (sampled from isotropic distrition).
        w: Weight of the current layer.
        """
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        uval = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        vval = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        uval.data = l2normalize(u.data)
        vval.data = l2normalize(v.data)
        wval = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_uval", uval)
        self.module.register_parameter(self.name + "_vval", vval)
        self.module.register_parameter(self.name + "_wval", wval)


    def forward(self, *args):
        """
        No need to change. Update weights using spectral normalization.
        """
        self._update_u_v()
        return self.module.forward(*args)
