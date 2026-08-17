import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, Categorical


def restore_parameters(model, best_model):

    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


class MaskLayer(nn.Module):


    def __init__(self, mask_size, append=True):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, mask):
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class MaskLayer2d(nn.Module):


    def __init__(self, mask_width, patch_size, append=False):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width ** 2


        self.patch_size = patch_size
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError('patch_size should be int >= 1')

    def forward(self, x, mask):

        if len(mask.shape) == 2:
            mask = mask.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(mask.shape) != 4:
            raise ValueError(f'cannot determine how to reshape mask with shape = {mask.shape}')


        mask = self.upsample(mask)
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class MaskLayerGrouped(nn.Module):


    def __init__(self, group_matrix, append=True):

        assert torch.all(group_matrix.sum(dim=0) == 1)
        assert torch.all((group_matrix == 0) | (group_matrix == 1))


        super().__init__()
        self.register_buffer('group_matrix', group_matrix.float())
        self.append = append
        self.mask_size = len(group_matrix)

    def forward(self, x, m):
        out = x * (m @ self.group_matrix)
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class StaticMaskLayer1d(torch.nn.Module):


    def __init__(self, inds):
        super().__init__()
        self.inds = inds

    def forward(self, x):
        return x[:, self.inds]


class StaticMaskLayer2d(torch.nn.Module):


    def __init__(self, mask, patch_size):
        super().__init__()
        self.patch_size = patch_size
        mask = mask.float()


        if len(mask.shape) == 4:
            assert mask.shape[0] == 1
            assert mask.shape[1] == 1
        elif len(mask.shape) == 3:
            assert mask.shape[0] == 1
            mask = torch.unsqueeze(mask, 0)
        elif len(mask.shape) == 2:
            mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
        else:
            raise ValueError(f'unable to reshape mask with size {mask.shape}')
        assert mask.shape[-1] == mask.shape[-2]


        if patch_size == 1:
            mask = mask
        elif patch_size > 1:
            mask = torch.nn.Upsample(scale_factor=patch_size)(mask)
        else:
            raise ValueError('patch_size should be int >= 1')
        self.register_buffer('mask', mask)
        self.mask_size = self.mask.shape[2] * self.mask.shape[3]

    def forward(self, x):
        out = x * self.mask
        return out


def generate_uniform_mask(batch_size, num_features):

    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()


def get_entropy(pred):

    return Categorical(logits=pred).entropy()


def get_confidence(pred):

    return torch.max(pred.softmax(dim=1), dim=1).values


def ind_to_onehot(inds, n):

    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot


def make_onehot(x):

    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot


class ConcreteSelector(nn.Module):


    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:

            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()


class ConcreteMask(nn.Module):


    def __init__(self, num_features, num_select, group_matrix=None, append=False, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, num_features, dtype=torch.float32))
        self.append = append
        self.gamma = gamma
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer('group_matrix', group_matrix.float())

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class ConcreteMask2d(nn.Module):


    def __init__(self, width, patch_size, num_select, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, width ** 2, dtype=torch.float32))
        self.upsample = torch.nn.Upsample(scale_factor=patch_size)
        self.width = width
        self.patch_size = patch_size
        self.gamma = gamma

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        m = self.upsample(m.reshape(-1, 1, self.width, self.width))
        out = x * m
        return out


def get_mlp_network(d_in, d_out, hidden=128, dropout=0.3):
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out))
