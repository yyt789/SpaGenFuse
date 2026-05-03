import copy
from itertools import chain
from math import ceil
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import torch.distributions as D
import torch.nn.functional as F
import scanpy
import itertools
import scipy.sparse as sp
from anndata import AnnData
from anndata._core.sparse_dataset import SparseDataset

from scipy.sparse import coo_matrix
from . import layers
from .data import ArrayDataset, DataLoader, ParallelDataLoader, calculate_affinity
from .utils import EarlyStopping, LRScheduler, Tensorboard, autodevice, get_default_numpy_dtype, \
    Model, Trainer, TrainingPlugin, poe
import os
from typing import Any, List, Mapping, Optional, NoReturn, Tuple, Dict
import ignite
import torch
from ..utils import config, logged, get_chained_attr, AnyArray
from torch.cuda.amp import GradScaler


AUTO = -1
DATA_CONFIG = Mapping[str, Any]

DataTensors = Tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]
def select_encoder(prob_model: str) -> type:

    if prob_model in ("ZIN", "ZILN"):
        return layers.VanillaDataEncoder
    if prob_model == "Normal":
        return layers.GaussianDataEncoder
    if prob_model in ("NB", "ZINB"):
        return layers.NBDataEncoder
    raise ValueError("Invalid `prob_model`!")
def select_decoder(prob_model: str) -> type:

    if prob_model == "Normal":
        return layers.GaussianDataDecoder
    if prob_model == "ZILN":
        return layers.ZILNDataDecoder
    if prob_model == "NB":
        return layers.NBDataDecoder

    raise ValueError("Invalid `prob_model`!")
@logged
class AnnDataset():

    def __init__(
            self, adatas: List[AnnData], data_configs: List[DATA_CONFIG],
            mode: str = "train", n_neighbors=10, sparse=False,  # getitem_size: int = 1
    ) -> None:

        if mode not in ("train", "eval"):
            raise ValueError("Invalid `mode`!")
        self.n_neighbors = n_neighbors
        self.sparse = sparse
        self.mode = mode
        self.adatas = adatas
        self.data_configs = data_configs

    @property
    def adatas(self) -> List[AnnData]:

        return self._adatas

    @property
    def data_configs(self) -> List[DATA_CONFIG]:

        return self._data_configs

    @adatas.setter
    def adatas(self, adatas: List[AnnData]) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        if min(self.sizes) == 0:
            raise ValueError("Empty dataset is not allowed!")
        for i in range(len(self.sizes) - 1):
            if self.sizes[i] != self.sizes[i + 1]:
                raise ValueError("Empty dataset is not allowed!")
        self.size = self.sizes[0]
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs: List[DATA_CONFIG]) -> None:
        if len(data_configs) != len(self.adatas):
            raise ValueError(
                "Number of data configs must match "
                "the number of datasets!"
            )
        self.adj, self.extracted_data = self._extract_data(
            data_configs)

        self._data_configs = data_configs

    def sparse_mx_to_torch_edge_list(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        edge_list = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        return edge_list

    def _extract_data(self, data_configs: List[DATA_CONFIG]) -> Tuple[
        List[pd.Index], Tuple[
            List[AnyArray], List[AnyArray], List[AnyArray],
            List[AnyArray], List[AnyArray]
        ]
    ]:
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)

    def _extract_data_train(self, data_configs: List[DATA_CONFIG]) -> Tuple[
        List[pd.Index], Tuple[
            List[AnyArray], List[AnyArray], List[AnyArray],
            List[AnyArray], List[AnyArray]
        ]
    ]:
        adj = [
            self._extract_adj(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = [
            self._extract_xbch(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xlbl = [
            self._extract_xlbl(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xdwt = [
            self._extract_xdwt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        return adj, (x, xalt, xbch, xlbl, xdwt)

    def _extract_data_eval(self, data_configs: List[DATA_CONFIG]) -> Tuple[
        List[pd.Index], Tuple[
            List[AnyArray], List[AnyArray], List[AnyArray],
            List[AnyArray], List[AnyArray]
        ]
    ]:
        default_dtype = get_default_numpy_dtype()
        adj = [
            self._extract_adj(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xalt = [
            self._extract_xalt(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config, xalt_ in zip(self.adatas, data_configs, xalt)
        ]
        xbch = xlbl = [
            np.empty((adata.shape[0], 0), dtype=int)
            for adata in self.adatas
        ]
        xdwt = [
            np.empty((adata.shape[0], 0), dtype=default_dtype)
            for adata in self.adatas
        ]
        return adj, (x, xalt, xbch, xlbl, xdwt)

    def _extract_x(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):

            adata = adata[:, features]

        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )

            x = adata.layers[use_layer]
        else:
            x = adata.obsm[data_config['use_rep']]
        if x.dtype.type is not default_dtype:
            if isinstance(x, (h5py.Dataset, SparseDataset)):
                raise RuntimeError(
                    f"User is responsible for ensuring a {default_dtype} dtype "
                    f"when using backed data!"
                )
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr()
        return x

    def _extract_xalt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_rep = data_config["use_rep"]
        rep_dim = data_config["rep_dim"]
        if use_rep:
            if use_rep not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_rep}' "
                    f"cannot be found in input data!"
                )
            xalt = adata.obsm[use_rep].astype(default_dtype)

            return xalt
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xbch(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_batch = data_config["use_batch"]
        batches = data_config["batches"]
        if use_batch:
            if use_batch not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_batch}' "
                    f"cannot be found in input data!"
                )
            return batches.get_indexer(adata.obs[use_batch])
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xlbl(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_cell_type = data_config["use_cell_type"]
        cell_types = data_config["cell_types"]
        if use_cell_type:
            if use_cell_type not in adata.obs:
                raise ValueError(
                    f"Configured cell type '{use_cell_type}' "
                    f"cannot be found in input data!"
                )
            return cell_types.get_indexer(adata.obs[use_cell_type])
        return -np.ones(adata.shape[0], dtype=int)

    def _extract_xdwt(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        use_dsc_weight = data_config["use_dsc_weight"]
        if use_dsc_weight:
            if use_dsc_weight not in adata.obs:
                raise ValueError(
                    f"Configured discriminator sample weight '{use_dsc_weight}' "
                    f"cannot be found in input data!"
                )
            xdwt = adata.obs[use_dsc_weight].to_numpy().astype(default_dtype)
            xdwt /= xdwt.sum() / xdwt.size
        else:
            xdwt = np.ones(adata.shape[0], dtype=default_dtype)
        return xdwt

    def _extract_adj(self, adata: AnnData, data_config: DATA_CONFIG) -> pd.Index:
        features = data_config["features"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]

        if self.n_neighbors is None and self.sparse:
            self.n_neighbors = 100
        adj = calculate_affinity(adata.X, sig=30, sparse=self.sparse, neighbors=self.n_neighbors)

        return adj
@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_gs: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_uid: Optional[str] = None
) -> None:

    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()

    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
        adata.layers[use_layer][adata.layers[use_layer] < 0] = 0
    else:
        data_config["use_layer"] = None
        adata.X[adata.X < 0] = 0
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    if use_uid:
        if use_uid not in adata.obs:
            raise ValueError("Invalid `use_uid`!")
        data_config["use_uid"] = use_uid
    else:
        data_config["use_uid"] = None
    adata.uns[config.ANNDATA_KEY] = data_config

class unispa(torch.nn.Module):

    def __init__(
            self,
            x2u: Mapping[str, layers.DataEncoder],
            u2z: Mapping[str, layers.DataEncoder],
            z2u: Mapping[str, layers.DataDecoder],
            u2x: Mapping[str, layers.DataDecoder],
            du: layers.Discriminator,
            du_gen: Mapping[str, layers.Discriminator], prior: layers.Prior,
            domains: dict
    ) -> None:
        super().__init__()
        if not set(x2u.keys()) == set(u2x.keys()) != set():
            raise ValueError(
                "`x2u`, `u2x`, `idx` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(x2u.keys())

        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2z = torch.nn.ModuleDict(u2z)
        self.z2u = torch.nn.ModuleDict(z2u)
        self.u2x = torch.nn.ModuleDict(u2x)

        self.du_gen = torch.nn.ModuleDict(du_gen)
        self.du = du
        self.prior = prior
        self.domains = domains
        self.device = autodevice()

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

class UniSpa(unispa):

    def __init__(
            self,
            x2u: Mapping[str, layers.DataEncoder],
            u2z: Mapping[str, layers.DataEncoder],
            z2u: Mapping[str, layers.DataDecoder],
            u2x: Mapping[str, layers.DataDecoder],
            du: layers.Discriminator, du_gen: Mapping[str, layers.Discriminator_gen], prior: layers.Prior,
            domains: dict,
            u2c: Optional[layers.Classifier] = None
    ) -> None:
        super().__init__(x2u, u2z, z2u, u2x, du, du_gen, prior, domains)
        self.u2c = u2c.to(self.device) if u2c else None

@logged
class UniSpaTrainer(Trainer):
    BURNIN_NOISE_EXAG: float = 1.5
    def __init__(
            self, net: UniSpa, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, lam_sc: float = 0.04, normalize_u: bool = None,
            domain_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, sparse: bool = False, **kwargs
    ) -> None:

        required_kwargs = (
            "lam_data", "lam_kl", "lam_graph", "lam_align",
            "domain_weight", "optim", "lr"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        super().__init__(net)

        self.required_losses = []
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo", f"x_{k}_sc_loss_u2z"]  # --------now
        self.required_losses += ["dsc_loss", "vae_loss", "gen_loss", "du_gen_loss_sum", "consistency_loss"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_sc = lam_sc
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        if min(domain_weight.values()) < 0:
            raise ValueError("Domain weight must be non-negative!")
        normalizer = sum(domain_weight.values()) / len(domain_weight)  # normalizer=1
        self.domain_weight = {k: v / normalizer for k, v in domain_weight.items()}

        self.lr = lr
        self.vae_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.x2u.parameters(),
                self.net.u2x.parameters(),
                self.net.u2z.parameters(),
                self.net.z2u.parameters()
            ), lr=self.lr, **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.du.parameters(),
                self.net.du_gen.parameters()),
            lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None

        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        self.freeze_u = False

        self.scaler = GradScaler()
        self.sparse = sparse

    @property
    def freeze_u(self) -> bool:

        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def sc_loss(self, A, Y):
        if not self.sparse:
            return (torch.triu(torch.cdist(Y, Y)) * torch.triu(A)).mean()
        else:
            row = A.coalesce().indices()[0]
            col = A.coalesce().indices()[1]
            rows1 = Y[row]
            rows2 = Y[col]
            dist = torch.norm(rows1 - rows2, dim=1)
            return (dist * A.coalesce().values()).mean()

    def calc_consistency_loss(self, z_uni: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_uni_stack = torch.stack(list(z_uni.values()),
                                  dim=0)

        z_uni_mean = z_uni_stack.mean(0, keepdim=True)

        consistency_loss = ((z_uni_stack - z_uni_mean) ** 2).sum() / z_uni_stack.size(1)  # Normalize by batch size
        return consistency_loss

    def generate_unified_latent(
            self,
            z_x_mu: Dict[str, torch.Tensor],
            z_x_stddev: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        z_uni = {}
        for modality, z_x_mu_mod in z_x_mu.items():

            combined_z = poe([z_x_mu_mod], [z_x_stddev[modality]])

            z_uni[modality] = combined_z.rsample()

        return z_uni

    def format_data(self, data: dict) -> DataTensors:

        device = self.net.device
        keys = self.net.keys

        x = data['x']
        xalt = data['xalt']
        xbch = data['xbch']
        xlbl = data['xlbl']
        xdwt = data['xdwt']
        adj = data['adj']

        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xalt = {
            k: xalt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        adj = {
            k: adj[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        # edge_index = edge_index.to(device, non_blocking=True)  # GCN

        # return x, xalt, xbch, xlbl, xdwt, xflag, edge_index  # GCN
        return x, xalt, xbch, xlbl, xdwt, xflag, adj

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net

        x, xalt, xbch, xlbl, xdwt, xflag, adj = data  # mlp


        xalt1 = xalt  # 2

        u, u1, z, l, x_gen, x_gen_cat, x_gen_flag_cat, usamp1 = {}, {}, {}, {}, {}, {}, {}, {}
        # -------------------------------- forward pass -------------------------
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xalt1[k], lazy_normalizer=dsc_only)  # mlp
        usamp = {k: u[k].rsample() for k in net.keys}

        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        mus, stds = [], []
        for k in net.keys:
            z[k] = net.u2z[k](u[k].mean)
            mus = mus + [z[k].mean]
            stds = stds + [z[k].stddev]
        combined_z = poe(mus, stds)

        for k in net.keys:
            u1[k] = net.z2u[k](combined_z.rsample())
            usamp1[k] = u1[k].rsample()
            x_gen[k] = net.u2x[k](
                usamp1[k], xbch[k], l[k]
            )
            x_gen_cat[k] = torch.cat([x_gen[k].sample(), x[k]])
            x_gen_flag_cat[k] = torch.cat([torch.zeros_like(xflag[k]), torch.ones_like(xflag[k])])

        # -------------------------------- calc_consistency_loss ------------------
        z_x_mu = {}
        z_x_stddev = {}
        for k, (mu, std) in zip(net.keys, zip(mus, stds)):
            z_x_mu[k] = mu
            z_x_stddev[k] = std
        z_uni = self.generate_unified_latent(z_x_mu, z_x_stddev)
        consistency_loss = self.calc_consistency_loss(z_uni)
        # -------------------------------- 计算真假判别器损失 -------------------------
        dsc_gen_loss = {
            k: (F.cross_entropy(net.du_gen[k](x_gen_cat[k]), x_gen_flag_cat[k], reduction="none")).mean()  # .sum()
            for k in net.keys
        }
        du_gen_loss_sum = sum(dsc_gen_loss[k] for k in net.keys)

        # -------------------------------- 计算谱聚类损失 -------------------------

        sc_loss_u2z = {
            k: self.sc_loss(adj[k], z[k].mean) for k in net.keys
        }

        # -------------------------------- 计算域判别器损失 -------------------------
        u_cat = torch.cat([z[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:

            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0],))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()

        # -------------------------------- 组成判别器损失 -------------------------
        if dsc_only:

            return {"dsc_loss": self.lam_align * du_gen_loss_sum}


        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        # -------------------------------- 计算重构损失 -------------------------
        x_u1_nll = {
            k: -net.u2x[k](
                usamp1[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }

        # -------------------------------- 计算重构损失(for elbo) -------------------------
        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        # -------------------------------- 计算kl散度(for elbo) -------------------------
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        # -------------------------------- 计算kl散度 -------------------------
        means = sum(u[k].mean for k in net.keys) / len(net.keys)
        scale = sum(u[k].stddev for k in net.keys) / len(net.keys)
        temp_D = D.Normal(means, scale)
        z_kl = {
            k: D.kl_divergence(
                z[k], temp_D
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        # -------------------------------- 计算证据下界(elbo) -------------------------
        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }

        # -------------------------------- 对模态损失求和 -------------------------
        x_elbo_sum = sum(self.domain_weight[k] * x_elbo[k] for k in net.keys)
        z_kl_sum = sum(self.domain_weight[k] * z_kl[k] for k in net.keys)
        x_u1_nll_sum = sum(self.domain_weight[k] * x_u1_nll[k] for k in net.keys)
        sc_loss_u2z_sum = sum(self.domain_weight[k] * sc_loss_u2z[k] for k in net.keys)


        # -------------------------------- 组成vae_loss -------------------------

        vae_loss = (self.lam_data * (x_elbo_sum + x_u1_nll_sum)
                    + 0.04 * z_kl_sum + 0.04 * consistency_loss) + 0.04 * sc_loss_u2z_sum


        # -------------------------------- 组成生成器损失 -------------------------

        gen_loss = vae_loss - self.lam_align * du_gen_loss_sum  #################### 0.32

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss, "du_gen_loss_sum": du_gen_loss_sum
            , "consistency_loss": consistency_loss
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k],
                f"x_{k}_sc_loss_u2z": sc_loss_u2z[k],
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def compute_losses_first(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net

        x, xalt, xbch, xlbl, xdwt, xflag, adj = data


        u, z, l = {}, {}, {}
        for k in net.keys:

            u[k], l[k] = net.x2u[k](x[k], xalt[k], lazy_normalizer=dsc_only)  # mlp
        usamp = {k: u[k].rsample() for k in net.keys}

        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}

        prior = net.prior()

        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.domain_weight[k] * x_elbo[k] for k in net.keys)

        vae_loss = self.lam_data * x_elbo_sum
        gen_loss = vae_loss

        losses = {
            "dsc_loss": torch.tensor(0.0, device=self.net.device), "vae_loss": vae_loss, "gen_loss": gen_loss,
            "du_gen_loss_sum": torch.tensor(0.0, device=self.net.device),
            "consistency_loss": torch.tensor(0.0, device=self.net.device)

        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k],
                f"x_{k}_sc_loss_u2z": torch.tensor(0.0, device=self.net.device),
            })
        if net.u2c:
            losses["sup_loss"] = torch.tensor(0.0, device=self.net.device)
        return losses

    def train_step(
            self, engine: ignite.engine.Engine, data: dict
    ) -> Mapping[str, torch.Tensor]:

        self.net.train()

        data = self.format_data(data)

        epoch = engine.state.epoch
        if self.safe_burnin:
            for i in range(2):
                losses = self.compute_losses(data, epoch, dsc_only=True)
                self.net.zero_grad(set_to_none=True)
                losses["dsc_loss"].backward()  # Already scaled by lam_align
                self.dsc_optim.step()


            losses = self.compute_losses(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()
            self.vae_optim.step()
            return losses
        else:
            losses = self.compute_losses_first(data, epoch)
            self.net.zero_grad(set_to_none=True)
            losses["gen_loss"].backward()
            self.vae_optim.step()
            return losses

    @torch.no_grad()
    def val_step(
            self, engine: ignite.engine.Engine, data: dict
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)

    def fit(
            self, data: AnnDataset, val_split: float = None,
            data_batch_size: int = None, graph_batch_size: int = None,
            align_burnin: int = None, safe_burnin: bool = True,
            max_epochs: int = None, patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            wait_n_lrs: Optional[int] = None,
            random_seed: int = None, directory: Optional[os.PathLike] = None,
            plugins: Optional[List[TrainingPlugin]] = None
    ) -> None:

        required_kwargs = (
            "val_split", "data_batch_size", "graph_batch_size",
            "align_burnin", "max_epochs", "random_seed"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )

        def to_dense_tensor(arr, dtype):
            if scipy.sparse.issparse(arr):
                arr = arr.toarray()

            arr = np.asarray(arr, dtype=np.float32 if dtype == torch.float32 else np.float64)
            return torch.as_tensor(arr, dtype=dtype)

        x, xalt, xbch, xlbl, xdwt = data.extracted_data
        default_dtype = torch.get_default_dtype()
        if not data.sparse:
            adj_tensor_list = [torch.as_tensor(i, dtype=default_dtype) for i in data.adj]
        else:
            adj = [coo_matrix(i) for i in data.adj]
            indices = [torch.as_tensor(np.vstack((i.row, i.col)), dtype=torch.long) for i in adj]
            values = [torch.as_tensor(i.data, dtype=default_dtype) for i in adj]
            shapes = [torch.Size(i.shape) for i in adj]
            adj_tensor_list = [torch.sparse_coo_tensor(indices[k], values[k], shapes[k]) for k in range(len(adj))]

        train_loader = [{
            "x": [to_dense_tensor(arr, default_dtype) for arr in x],
            "xalt": [torch.as_tensor(arr, dtype=default_dtype) for arr in xalt],
            "xbch": [torch.as_tensor(arr, dtype=torch.long) for arr in xbch],
            "xlbl": [torch.as_tensor(arr, dtype=torch.long) for arr in xlbl],
            "xdwt": [torch.as_tensor(arr, dtype=default_dtype) for arr in xdwt],
            "adj": adj_tensor_list,
        }]


        print("--------------------------where train_loader")
        for i in range(len(x)):
            print(f"x[{i}].shape: {train_loader[0]['x'][i].shape}")
            print(f"xalt[{i}].shape: {train_loader[0]['xalt'][i].shape}")
            print(f"xbch[{i}].shape: {train_loader[0]['xbch'][i].shape}")
            print(f"xlbl[{i}].shape: {train_loader[0]['xlbl'][i].shape}")
            print(f"xdwt[{i}].shape: {train_loader[0]['xdwt'][i].shape}")
            print(f"adj[{i}].shape: {train_loader[0]['adj'][i].shape}")

        val_loader = None

        self.align_burnin = align_burnin
        self.safe_burnin = safe_burnin

        default_plugins = [Tensorboard()]
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin if safe_burnin else 0
            ))
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin if safe_burnin else 0,
                wait_n_lrs=wait_n_lrs or 0
            ))
        plugins = default_plugins + (plugins or [])
        try:
            super().fit(
                train_loader, val_loader=val_loader,
                max_epochs=max_epochs, random_seed=random_seed,
                directory=directory, plugins=plugins
            )
        finally:

            self.align_burnin = None
            self.safe_burnin = None

    def get_losses(
            self, data: ArrayDataset,
            data_batch_size: int = None,
            random_seed: int = None
    ) -> Mapping[str, float]:
        required_kwargs = ("data_batch_size", "graph_batch_size", "random_seed")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        data.getitem_size = data_batch_size

        data.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        loader = ParallelDataLoader(
            DataLoader(
                data, batch_size=1, shuffle=True, drop_last=False,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            )
        )

        try:
            losses = super().get_losses(loader)
        finally:
            data.clean()
            self.eidx = None
            self.enorm = None
            self.esgn = None

        return losses

    def state_dict(self) -> Mapping[str, Any]:
        return {
            **super().state_dict(),
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
        super().load_state_dict(state_dict)

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )
@logged
class UniSpaModel(Model):
    NET_TYPE = UniSpa
    TRAINER_TYPE = UniSpaTrainer

    GRAPH_BATCHES: int = 32
    ALIGN_BURNIN_PRG: float = 8.0
    MAX_EPOCHS_PRG: float = 48.0
    PATIENCE_PRG: float = 4.0
    REDUCE_LR_PATIENCE_PRG: float = 2.0

    def __init__(
            self, adatas: Mapping[str, AnnData], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0, sparse=False,
    ) -> None:
        self.sparse = sparse
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.domains, x2u, u2z, z2u, u2x, du_gen, all_ct = {}, {}, {}, {}, {}, {}, set()
        for k, adata in adatas.items():

            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            print(f"[debug]{k}", len(data_config["features"]))
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )

            x2u[k] = select_encoder(data_config["prob_model"])(
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )

            u2z[k] = layers.ZEncoder(50, 50)

            z2u[k] = layers.ZDecoder(50, 50)
            du_gen[k] = layers.Discriminator_gen(
                in_features=len(data_config["features"]) if data_config["use_layer"] else data_config["rep_dim"],

                out_features=2, n_batches=0,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )

            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = select_decoder(data_config["prob_model"])(
                out_features=len(data_config["features"]) if data_config["use_layer"] else data_config["rep_dim"],

                n_batches=max(data_config["batches"].size, 1)
            )

            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.domains[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for domain in self.domains.values():
            domain["cell_types"] = all_ct
        if shared_batches:
            all_batches = [domain["batches"] for domain in self.domains.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = layers.Discriminator(
            latent_dim, len(self.domains), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )

        prior = layers.Prior()
        super().__init__(
            x2u, u2z, z2u, u2x, du, du_gen, prior, domains=self.domains
        )

    def freeze_cells(self) -> None:

        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:

        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "UniSpaModel", submodule: Optional[str] = None
    ) -> None:

        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(
            self, lam_data: float = 1.0,
            lam_sc: float = 0.04,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            normalize_u: bool = False,
            domain_weight: Optional[Mapping[str, float]] = None,
            lr: float = 1e-3, **kwargs
    ) -> None:

        if domain_weight is None:
            domain_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl, lam_sc=lam_sc,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            normalize_u=normalize_u, domain_weight=domain_weight,
            optim="RMSprop", lr=lr, sparse=self.sparse, **kwargs
        )

    def fit(
            self, adatas: Mapping[str, AnnData],
            edge_weight: str = "weight", edge_sign: str = "sign",
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None
    ) -> None:

        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.domains[key] for key in self.net.keys],
            mode="train", sparse=self.sparse
        )
        print(f"################data.size:{data.size}###############")


        batch_per_epoch = data.size / data_batch_size

        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def get_losses(
            self, adatas: Mapping[str, AnnData], data_batch_size: int = 128
    ) -> Mapping[str, np.ndarray]:

        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.domains[key] for key in self.net.keys],
            mode="train"
        )

        return super().get_losses(
            data, data_batch_size=data_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, n_sample: Optional[int] = None
    ) -> np.ndarray:

        self.net.eval()
        encoder = self.net.x2u[key]
        u2z = self.net.u2z[key]
        data = AnnDataset(
            [adata], [self.domains[key]],
            mode="eval",
        )

        def to_dense_tensor(arr, dtype):
            if scipy.sparse.issparse(arr):
                arr = arr.toarray()

            arr = np.asarray(arr, dtype=np.float32 if dtype == torch.float32 else np.float64)
            return torch.as_tensor(arr, dtype=dtype)

        x, xalt, xbch, xlbl, xdwt = data.extracted_data
        default_dtype = torch.get_default_dtype()

        x = to_dense_tensor(x[0], default_dtype).to(self.net.device, non_blocking=True)
        xalt = torch.as_tensor(xalt[0], dtype=default_dtype).to(self.net.device, non_blocking=True)


        with torch.no_grad():

            u_dist = encoder(x, xalt, lazy_normalizer=True)[0]  # mlp
            z_dist = u2z(u_dist.mean)

            if n_sample:
                return z_dist.sample((n_sample,)).cpu().permute(1, 0, 2).numpy()
            else:
                return z_dist.mean.cpu().numpy(), z_dist.stddev.cpu().numpy()

    @torch.no_grad()
    def generate_cross(
            self, key1: str, key2: str, adata: AnnData, adata_other: AnnData, batch_size: int = 128
    ) -> np.ndarray:

        self.net.eval()
        encoder = self.net.x2u[key1]
        encoder_other = self.net.x2u[key2]

        u2z = self.net.u2z[key1]
        z2u = self.net.z2u[key2]
        u2x = self.net.u2x[key2]
        data = AnnDataset(
            [adata], [self.domains[key1]],
            mode="eval", getitem_size=batch_size
        )
        data_other = AnnDataset(
            [adata_other], [self.domains[key2]],
            mode="eval", getitem_size=batch_size
        )

        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        data_loader_other = DataLoader(
            data_other, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        result = []

        l_other = torch.Tensor().cuda()

        for x, xalt, *_ in data_loader_other:
            xalt = xalt[:, :-50]
            u_other, l_other_1 = encoder_other(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )
            l_other = torch.cat((l_other, l_other_1))

        l_other = torch.mean(l_other)

        for x, xalt, *_ in data_loader:
            xalt = xalt[:, :-50]
            u, l = encoder(
                x.to(self.net.device, non_blocking=True),
                xalt.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )

            z = u2z(u.mean)
            u1 = z2u(z.mean)
            b = np.zeros(len(l), dtype=int)
            l = l / torch.mean(l) * l_other

            u1samp = u1.rsample()
            x_out = u2x(u1samp, b, l)

            result.append(x_out.sample().cpu())

        return torch.cat(result).numpy()

    @torch.no_grad()
    def generate_multiSim(
            self, adatas: Mapping[str, AnnData], obs_from: str, name: str, num: int, batch_size: int = 128
    ) -> []:

        self.net.eval()

        l_s = []
        z_s = torch.Tensor().cuda()
        z_d_s = torch.Tensor().cuda()

        for key, adata in adatas.items():
            x2u = self.net.x2u[key]
            u2z = self.net.u2z[key]
            adata_sub = adata[adata.obs[obs_from].isin([name])]
            data = AnnDataset(
                [adata_sub], [self.domains[key]],
                mode="eval", getitem_size=len(adata_sub.obs)
            )
            data_loader = DataLoader(
                data, batch_size=1, shuffle=False,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                persistent_workers=False
            )

            l_s_t = []

            for x, xalt, *_ in data_loader:
                xalt = xalt[:, :-50]

                u, l = x2u(
                    x.to(self.net.device, non_blocking=True),
                    xalt.to(self.net.device, non_blocking=True),
                    lazy_normalizer=True
                )
                z = u2z(u.mean)

                l = torch.mean(l.cpu())

                z_t = torch.mean(z.mean, dim=0, keepdim=True)
                z_d = torch.mean(z.stddev, dim=0, keepdim=True)

                l_s_t.append(l)
                z_s = torch.cat((z_s, z_t))
                z_d_s = torch.cat((z_d_s, z_d))

            l_s.append(np.mean(l_s_t))

        z_s_m = torch.mean(z_s, dim=0, keepdim=True)
        z_d_s_m = torch.mean(z_d_s, dim=0, keepdim=True)

        g = 0
        result_s = {}
        result_t = []
        for key, adata in adatas.items():
            result_s[key] = torch.Tensor()
        z = D.Normal(z_s_m, z_d_s_m)
        for i in range(num):
            u1samp = z.rsample()
            g = 0

            for key, adata in adatas.items():
                z2u = self.net.z2u[key]
                u2x = self.net.u2x[key]
                u = z2u(u1samp)
                l = l_s[g]
                b = 0
                g = g + 1
                x_out = u2x(u.mean, b, l)
                result_s[key] = torch.cat((result_s[key], x_out.sample().cpu()))

        for key, adata in adatas.items():
            result = result_s[key].numpy()
            adata_s = adata[:, adata.var.query("highly_variable").index.to_numpy().tolist()]
            result_a = scanpy.AnnData(result, var=adata_s.var)
            result_t.append(result_a)

        return result_t

    def __repr__(self) -> str:
        return (
            f"UniSpa model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )
