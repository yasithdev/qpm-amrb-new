from typing import Tuple
import warnings

import lightning.pytorch as pl
import pandas as pd
import PIL.Image
import seaborn as sns
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchmetrics.aggregation import CatMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        labels: list[str],
        cat_k: int,
        optim_lr: float,
        with_encoder: bool = True,
        with_classifier: bool = True,
        with_decoder: bool = True,
        with_permutations: bool = False,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.cat_k = cat_k
        self.optim_lr = optim_lr
        self.with_encoder = with_encoder
        self.with_classifier = with_classifier
        self.with_decoder = with_decoder
        self.with_permutations = with_permutations

    def define_model(self):
        raise NotImplementedError()

    def compute_losses(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def log_losses(
        self,
        stage: str,
        losses: dict[str, torch.Tensor],
    ):
        for name, value in losses.items():
            # only logging rank-zero process to avoid deadlocks
            self.log(f"{stage}_{name}", value, prog_bar=True, rank_zero_only=True, sync_dist=True)

    def define_metrics(self):
        K = self.cat_k
        for stage in ["train", "val", "test", "predict"]:
            # common metrics
            setattr(self, f"{stage}_batch_x_true", CatMetric())
            setattr(self, f"{stage}_batch_y_true", CatMetric())
            # classifier metrics
            if self.with_classifier:
                setattr(self, f"{stage}_accuracy", MulticlassAccuracy(num_classes=K, top_k=1))
                setattr(self, f"{stage}_accuracy_top2", MulticlassAccuracy(num_classes=K, top_k=2))
                setattr(self, f"{stage}_confusion_matrix", MulticlassConfusionMatrix(num_classes=K))
                setattr(self, f"{stage}_uncertainty", CatMetric())
                setattr(self, f"{stage}_correctness", CatMetric())
                setattr(self, f"{stage}_batch_y_pred", CatMetric())
                setattr(self, f"{stage}_batch_y_ucty", CatMetric())
            # permutation metrics
            if self.with_permutations:
                setattr(self, f"{stage}_uncertainty_rand", CatMetric())
                setattr(self, f"{stage}_correctness_rand", CatMetric())
            # decoder metrics
            if self.with_decoder:
                setattr(self, f"{stage}_decoder_mse", MeanSquaredError())
                setattr(self, f"{stage}_batch_x_pred", CatMetric())

    def reset_metrics(self, stage: str):
        getattr(self, f"{stage}_batch_x_true").reset()
        getattr(self, f"{stage}_batch_y_true").reset()
        if self.with_classifier:
            getattr(self, f"{stage}_accuracy").reset()
            getattr(self, f"{stage}_accuracy_top2").reset()
            getattr(self, f"{stage}_confusion_matrix").reset()
            getattr(self, f"{stage}_uncertainty").reset()
            getattr(self, f"{stage}_correctness").reset()
            getattr(self, f"{stage}_batch_y_pred").reset()
            getattr(self, f"{stage}_batch_y_ucty").reset()
        # permutation metrics
        if self.with_permutations:
            getattr(self, f"{stage}_uncertainty_rand").reset()
            getattr(self, f"{stage}_correctness_rand").reset()
        if self.with_decoder:
            getattr(self, f"{stage}_decoder_mse").reset()
            getattr(self, f"{stage}_batch_x_pred").reset()

    def update_metrics(
        self,
        stage: str,
        batch_idx: int,
        metrics: dict[str, torch.Tensor],
    ):
        if "y_pred" in metrics and "y_true" in metrics:
            arg = metrics["y_pred"].eq(metrics["y_true"])
            getattr(self, f"{stage}_correctness")(arg)
        if "y_ucty" in metrics:
            arg = metrics["y_ucty"]
            getattr(self, f"{stage}_uncertainty")(arg)
        if "y_prob" in metrics and "y_true" in metrics:
            args = metrics["y_prob"], metrics["y_true"]
            getattr(self, f"{stage}_accuracy")(*args)
            getattr(self, f"{stage}_accuracy_top2")(*args)
            getattr(self, f"{stage}_confusion_matrix")(*args)

        # permutation metrics
        if "y_pred_rand" in metrics and "y_true_rand" in metrics:
            arg = metrics["y_pred_rand"].eq(metrics["y_true_rand"])
            getattr(self, f"{stage}_correctness_rand")(arg)
        if "y_ucty_rand" in metrics:
            arg = metrics["y_ucty_rand"]
            getattr(self, f"{stage}_uncertainty_rand")(arg)

        if "x_pred" in metrics and "x_true" in metrics:
            args = metrics["x_pred"].flatten(start_dim=1), metrics["x_true"].flatten(start_dim=1)
            getattr(self, f"{stage}_decoder_mse")(*args)

        if batch_idx == 0:
            if "x_true" in metrics:
                getattr(self, f"{stage}_batch_x_true")(metrics["x_true"])
            if "y_true" in metrics:
                getattr(self, f"{stage}_batch_y_true")(metrics["y_true"])
            if "x_pred" in metrics:
                getattr(self, f"{stage}_batch_x_pred")(metrics["x_pred"])
            if "y_pred" in metrics:
                getattr(self, f"{stage}_batch_y_pred")(metrics["y_pred"])
            if "y_ucty" in metrics:
                getattr(self, f"{stage}_batch_y_ucty")(metrics["y_ucty"])

    def on_epoch_end(self, stage: str):
        # get wandb logger
        assert self.logger
        logger: WandbLogger = self.logger  # type: ignore
        K = self.cat_k

        # classifier related plots
        if self.with_classifier:
            # log accuracy
            acc: torch.Tensor = getattr(self, f"{stage}_accuracy").compute()
            acc2: torch.Tensor = getattr(self, f"{stage}_accuracy_top2").compute()
            self.log(f"{stage}_accuracy", acc, sync_dist=True)
            self.log(f"{stage}_accuracy_top2", acc2, sync_dist=True)
            # plot confusion matrix
            confusion_matrix: MulticlassConfusionMatrix = getattr(self, f"{stage}_confusion_matrix")
            cm_fig: Figure = confusion_matrix.plot(add_text=True, labels=self.labels[: self.cat_k])[0]  # type: ignore
            cm_fig.canvas.draw()
            cm_img = PIL.Image.frombytes("RGB", cm_fig.canvas.get_width_height(), cm_fig.canvas.tostring_rgb())  # type: ignore
            logger.log_image(f"{stage}_confusion_matrix", [cm_img])
            plt.close()
            # plot model calibration
            uncertainty: torch.Tensor = getattr(self, f"{stage}_uncertainty").compute()
            correctness: torch.Tensor = getattr(self, f"{stage}_correctness").compute().bool()
            ucty_T = pd.DataFrame({"ucty": uncertainty[correctness].tolist()})
            ucty_F = pd.DataFrame({"ucty": uncertainty[~correctness].tolist()})
            fig = plt.figure()
            if len(ucty_T) > 0:    
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    sns.kdeplot(ucty_T, x="ucty", fill=True, label="Correct Predictions")
            if len(ucty_F) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    sns.kdeplot(ucty_F, x="ucty", fill=True, label="Incorrect Predictions")
            plt.legend(loc="upper right")
            plt.title("Model Calibration")
            fig.canvas.draw()
            cd_img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())  # type: ignore
            logger.log_image(f"{stage}_calibration", [cd_img])
            plt.close()

            # plot null distribution from permutations
            if self.with_permutations:
                # TODO check if this working is correct
                uncertainty_gt: torch.Tensor = uncertainty.mean()
                correctness_gt: torch.Tensor = correctness.float().mean()
                uncertainty_rand: torch.Tensor = getattr(self, f"{stage}_uncertainty_rand").compute()  # (B, N-1)
                correctness_rand: torch.Tensor = getattr(self, f"{stage}_correctness_rand").compute()  # (B, N-1)
                dist_data = pd.DataFrame({
                    "correctness_rand": correctness_rand.float().mean(0).tolist(), # (N-1,)
                    "uncertainty_rand": uncertainty_rand.float().mean(0).tolist(), # (N-1,)
                })
                fig = plt.figure()
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    sns.histplot(dist_data, x="correctness_rand", fill=True, kde=True)
                plt.axvline(correctness_gt.item(), color='red')
                plt.title("Null Distribution (Accuracy)")
                fig.canvas.draw()
                cd_img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())  # type: ignore
                logger.log_image(f"{stage}_null_dist_acc", [cd_img])
                plt.close()
                fig = plt.figure()
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    sns.histplot(dist_data, x="uncertainty_rand", fill=True, kde=True)
                plt.axvline(uncertainty_gt.item(), color='red')
                plt.title("Null Distribution (Uncertainty)")
                fig.canvas.draw()
                cd_img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())  # type: ignore
                logger.log_image(f"{stage}_null_dist_ucty", [cd_img])
                plt.close()

        # decoder related plots
        if self.with_decoder:
            # log mse
            self.log(f"{stage}_decoder_mse", getattr(self, f"{stage}_decoder_mse").compute(), sync_dist=True)
            # prepare data for sample plots
            N = 10
            x_true = getattr(self, f"{stage}_batch_x_true").compute()
            x_pred = getattr(self, f"{stage}_batch_x_pred").compute()
            y_true = getattr(self, f"{stage}_batch_y_true").compute().int()
            y_true_text = [self.labels[k] for k in y_true]
            if self.with_classifier:
                y_pred = getattr(self, f"{stage}_batch_y_pred").compute().int()
                y_ucty = getattr(self, f"{stage}_batch_y_ucty").compute().flatten()
                y_pred_text = [f"{self.labels[a]}, ucty={b:.2f}" for a, b in zip(y_pred, y_ucty)]
            else:
                y_pred_text = y_true_text
            # plot samples
            logger.log_image(f"{stage}_batch_x_true", list(x_true)[:N], caption=y_true_text[:N])
            logger.log_image(f"{stage}_batch_x_pred", list(x_pred)[:N], caption=y_pred_text[:N])

        # reset metrics
        self.reset_metrics(stage)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        L_mb = self.compute_losses(batch, batch_idx, stage="train")
        return L_mb

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end(stage="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.compute_losses(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end(stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.compute_losses(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end(stage="test")

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.compute_losses(batch, batch_idx, stage="predict")

    def on_predict_epoch_end(self) -> None:
        self.on_epoch_end(stage="predict")
