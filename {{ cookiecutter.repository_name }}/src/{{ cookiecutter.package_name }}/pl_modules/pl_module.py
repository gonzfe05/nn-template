import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from {{ cookiecutter.package_name }}.data.datamodule import MetaData
from {{ cookiecutter.package_name }}.modules.losses import ContrastiveLoss

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model: nn.Module, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        # example
        assert self.metadata.task
        if self.metadata.task == "binary":
            metric = torchmetrics.Accuracy(self.metadata.task, threshold=self.metadata.threshold)
        elif self.metadata.task == "multiclass":
            num_classes = len(self.metadata.class_vocab.keys())
            metric = torchmetrics.Accuracy(self.metadata.task, num_classes=num_classes)
        elif self.metadata.task == "multilabel":
            metric = torchmetrics.Accuracy(self.metadata.task, threshold=self.metadata.threshold)
        else:
            raise ValueError(f"Unrecognized task value: '{self.metadata.task}'")
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x)

    def step(self, x, y) -> Mapping[str, Any]:
        # example
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


class MyContrastativeLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model: nn.Module, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        # example
        assert self.metadata.task
        if self.metadata.task == "binary":
            metric = torchmetrics.Accuracy(self.metadata.task, threshold=self.metadata.threshold)
        else:
            raise ValueError(f"Unrecognized task value: '{self.metadata.task}'")
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()
        self.model = model
        self.loss = ContrastiveLoss()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x0, x1)

    def step(self, i0, i1, y) -> Mapping[str, Any]:
        # example
        x0, x1 = self(i0, i1)
        loss = self.loss(x0, x1, y)
        dist, _ = self.loss.get_distance(x0, x1)
        return {"margin": dist, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x0, x1, y = batch
        step_out = self.step(x0, x1, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy((step_out["margin"] > 0).float(), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x0, x1, y = batch
        step_out = self.step(x0, x1, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy((step_out["margin"] > 0).float(), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x0, x1, y = batch
        step_out = self.step(x0, x1, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy((step_out["margin"] > 0).float(), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    metadata = MetaData({"0": 0, "1": 1}, cfg.data.datamodule.task)
    model = hydra.utils.instantiate(cfg.model.arch, _recursive_=False, num_classes=len(metadata.class_vocab))
    pl_module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module, _recursive_=False, metadata=metadata, model=model, **cfg.optim
    )
    print(pl_module)
    cfg.model.arch["_target_"] = cfg.model.arch["_target_"].replace("CNN", "SiameseNetwork")
    model = hydra.utils.instantiate(cfg.model.arch, _recursive_=False, num_classes=len(metadata.class_vocab))
    pl_module: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.module, _recursive_=False, metadata=metadata, model=model, **cfg.optim
    )
    print(pl_module)

if __name__ == "__main__":
    main()
