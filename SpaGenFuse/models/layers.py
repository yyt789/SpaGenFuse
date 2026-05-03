import collections
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn.functional as F


from ..utils import EPS

from .utils import ZILN, ZIN, ZINB

from torch_geometric.nn import GCNConv


class DataEncoder(torch.nn.Module):
    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", GCNConv(ptr_dim, h_dim, cached=False))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:

        raise NotImplementedError

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:

        raise NotImplementedError

    def forward(
            self, x: torch.Tensor, xalt: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:

        if xalt.numel():

            l = self.compute_l(x)
            ptr = xalt
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):


    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)



    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x


# x2u
class NBDataEncoder(torch.nn.Module):


    TOTAL_COUNT = 1e4

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        ptr_dim1 = ptr_dim
        for layer in range(self.h_depth):

            setattr(self, f"liner1_{layer}", torch.nn.Linear(ptr_dim1, h_dim))  # mlp
            setattr(self, f"act1_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn1_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout1_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim1 = h_dim
        self.loc1 = torch.nn.Linear(ptr_dim1, out_features)
        self.std_lin1 = torch.nn.Linear(ptr_dim1, out_features)

    def forward(
            self, x: torch.Tensor, xalt: torch.Tensor,


    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:


        ptr1 = xalt
        l = self.compute_l(x)

        for layer in range(self.h_depth):
            # ptr1 = getattr(self, f"GCNConv1_{layer}")(ptr1, edge_index)  # GCN
            ptr1 = getattr(self, f"liner1_{layer}")(ptr1)  # mlp
            ptr1 = getattr(self, f"act1_{layer}")(ptr1)
            ptr1 = getattr(self, f"bn1_{layer}")(ptr1)
            ptr1 = getattr(self, f"dropout1_{layer}")(ptr1)
        loc = self.loc1(ptr1)
        std = F.softplus(self.std_lin1(ptr1)) + EPS

        return D.Normal(loc, std), l

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


# x2u
class GaussianDataEncoder(torch.nn.Module):

    TOTAL_COUNT = 1e4

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        ptr_dim1 = ptr_dim

        for layer in range(self.h_depth):
            setattr(self, f"liner1_{layer}", torch.nn.Linear(ptr_dim1, h_dim))
            setattr(self, f"act1_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn1_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout1_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim1 = h_dim
        self.loc1 = torch.nn.Linear(ptr_dim1, out_features)
        self.std_lin1 = torch.nn.Linear(ptr_dim1, out_features)

    def forward(
            self, x: torch.Tensor, xalt: torch.Tensor,

    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:


        ptr1 = xalt
        l = self.compute_l(x)

        for layer in range(self.h_depth):

            ptr1 = getattr(self, f"liner1_{layer}")(ptr1)
            ptr1 = getattr(self, f"act1_{layer}")(ptr1)
            ptr1 = getattr(self, f"bn1_{layer}")(ptr1)
            ptr1 = getattr(self, f"dropout1_{layer}")(ptr1)
        loc = self.loc1(ptr1)
        std = F.softplus(self.std_lin1(ptr1)) + EPS

        return D.Normal(loc, std), l

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


# u2z
class ZEncoder(torch.nn.Module):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 1, h_dim: int = 16,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features

        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(
            self, x: torch.Tensor,  # edge_index: torch.Tensor,
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        ptr = x

        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


# z2u
class ZDecoder(torch.nn.Module):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 1, h_dim: int = 50,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features

        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        ptr = x

        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class DataDecoder(torch.nn.Module):


    def __init__(self, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:

        raise NotImplementedError  # pragma: no cover


class ZILNDataDecoder(DataDecoder):


    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.h_depth = 1
        ptr_dim = 50
        h_dim = out_features

        setattr(self, f"linear_", torch.nn.Linear(ptr_dim, h_dim))

    def forward(
            self, u: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        ptr = u

        ptr = getattr(self, f"linear_")(ptr)

        scale = F.softplus(self.scale_lin[b])
        loc = scale * ptr + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


# u2x
class NBDataDecoder(DataDecoder):


    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.h_depth = 1
        ptr_dim = 50
        h_dim = out_features


        setattr(self, f"linear_", torch.nn.Linear(ptr_dim, h_dim))

    def forward(
            self, u: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])  # (N,out_features)
        ptr = u

        ptr = getattr(self, f"linear_")(ptr)

        logit_mu = scale * ptr + self.bias[b]



        mu = F.softmax(logit_mu, dim=1) * l  # (N, out_features)

        log_theta = self.log_theta[b]

        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log()
        )


# u2x
class GaussianDataDecoder(torch.nn.Module):
    r"""
    Gaussian (Normal) data decoder

    Parameters
    ----------
    out_features: int
        Output dimensionality
    n_batches: int
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__()
        self.out_features = out_features
        self.n_batches = n_batches

        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_var = torch.nn.Parameter(torch.zeros(n_batches, out_features))  # log(variance)

        ptr_dim = 50  # input dim
        h_dim = out_features
        setattr(self, f"linear_", torch.nn.Linear(ptr_dim, h_dim))

    def forward(
            self, u: torch.Tensor,  # latent embedding: (N, ptr_dim)
            b: torch.Tensor,  # batch indices: (N,)
            l: torch.Tensor  # library size: (N, 1) or (N,)
    ) -> D.Normal:
        # Compute mean
        scale = F.softplus(self.scale_lin[b])  # (N, out_features)
        ptr = u
        ptr = getattr(self, f"linear_")(ptr)

        mu = scale * ptr + self.bias[b]  # (N, out_features)

        mu = mu * l  # broadcasting (N, 1)

        # Compute std from log-variance
        log_var = self.log_var[b]  # (N, out_features)
        std = (0.5 * self.log_var[b]).exp().clamp(min=EPS)  # (N, out_features)

        return D.Normal(loc=mu, scale=std)


class Discriminator(torch.nn.Sequential, torch.nn.Module):
    r"""
    Domain discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Discriminator_gen(torch.nn.Sequential, torch.nn.Module):
    r"""
    Domain discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return super().forward(x)


class Classifier(torch.nn.Linear):
    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


class Prior(torch.nn.Module):
    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)
