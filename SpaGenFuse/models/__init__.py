import os
from typing import Mapping
import numpy as np
from anndata import AnnData
from ..utils import logged, Kws
from .utils import Model
from .SpaGenFuse import (AUTO, UniSpaModel,
                         configure_dataset)
import torch


def load_model(fname: os.PathLike) -> Model:
    return Model.load(fname)

@logged
def fit_UniSpa(
        adatas: Mapping[str, AnnData], model: type = UniSpaModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None
) -> UniSpaModel:
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}

    fit_UniSpa.logger.info("Pretraining SpaGenFuse model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False}) 
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update(
        {"align_burnin": np.inf, "safe_burnin": False})  
    if "directory" in pretrain_fit_kws: 
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")


    pretrain = model(adatas, **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, **pretrain_fit_kws)

    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    fit_UniSpa.logger.info("Fine-tuning SpaGenFuse model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune")


    finetune = model(adatas, **init_kws)
    finetune.adopt_pretrained_model(pretrain)

    torch.cuda.empty_cache()  #
    finetune.compile(**compile_kws)
    finetune.fit(adatas, **finetune_fit_kws)

    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune
