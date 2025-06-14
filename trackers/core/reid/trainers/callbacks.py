import os
from typing import Any, Optional

import matplotlib.pyplot as plt


class BaseCallback:
    def on_train_batch_start(self, logs: dict, idx: int):
        pass

    def on_train_batch_end(self, logs: dict, idx: int):
        pass

    def on_train_epoch_end(self, logs: dict, epoch: int):
        pass

    def on_validation_batch_start(self, logs: dict, idx: int):
        pass

    def on_validation_batch_end(self, logs: dict, idx: int):
        pass

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        pass

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        pass

    def on_end(self):
        pass


class TensorboardCallback(BaseCallback):
    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[Any] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir,
            comment=comment,
            filename_suffix=filename_suffix,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
        )

    def on_train_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_train_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)

    def on_end(self):
        self.writer.flush()
        self.writer.close()


class WandbCallback(BaseCallback):
    def __init__(self, config: dict[str, Any]) -> None:
        import wandb

        self.run = wandb.init(config=config) if not wandb.run else wandb.run  # type: ignore

        self.run.define_metric("batch/step")
        self.run.define_metric("batch/train/loss", step_metric="batch/step")

        self.run.define_metric("epoch")
        self.run.define_metric("train/loss", step_metric="epoch")
        self.run.define_metric("validation/loss", step_metric="epoch")

    def on_train_batch_end(self, logs: dict, idx: int):
        logs["batch/step"] = idx
        self.run.log(logs)

    def on_train_epoch_end(self, logs: dict, epoch: int):
        logs["epoch"] = epoch
        self.run.log(logs)

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        logs["epoch"] = epoch
        self.run.log(logs)

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        self.run.log_model(
            path=checkpoint_path,
            name=f"checkpoint_{self.run.id}",
            aliases=[f"epoch-{epoch}", "latest"],
        )

    def on_end(self):
        self.run.finish()


class MatplotlibCallback(BaseCallback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.train_history: dict[str, list[tuple[int, float]]] = {}
        self.validation_history: dict[str, list[tuple[int, float]]] = {}

    def on_train_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.train_history.setdefault(key, []).append((idx, value))

    def on_train_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.train_history.setdefault(key, []).append((epoch, value))

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.validation_history.setdefault(key, []).append((epoch, value))

    def _plot_subplot(
        self,
        ax,
        title_prefix: str,
        base_metric_name: str,
        xlabel: str,
        train_keys: list[str],
        val_keys: Optional[list[str]] = None,
    ):
        train_data_points = []
        for key in train_keys:
            data = self.train_history.get(key)
            if data:  # Checks for None and non-empty list
                train_data_points = data
                break

        val_data_points = []
        if val_keys:
            for key in val_keys:
                data = self.validation_history.get(key)
                if data:  # Checks for None and non-empty list
                    val_data_points = data
                    break

        plotted_anything = False
        if train_data_points:
            x, y = zip(*train_data_points)
            ax.plot(x, y, label="train", marker=".", markersize=5, linewidth=1)
            plotted_anything = True

        if val_data_points:
            x_val, y_val = zip(*val_data_points)
            ax.plot(
                x_val,
                y_val,
                label="validation",
                marker=".",
                markersize=5,
                linewidth=1,
                linestyle="--",
            )
            plotted_anything = True

        formatted_base_name = " ".join(
            [item.capitalize() for item in base_metric_name.split("_")]
        )
        ax.set_title(f"{title_prefix} {formatted_base_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(formatted_base_name)

        if plotted_anything:
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("lightgray")

    def on_end(self):
        if not self.train_history and not self.validation_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

        # Plot 1: Top-left - Batch Triplet Accuracy
        self._plot_subplot(
            axes[0, 0],
            title_prefix="Batch",
            base_metric_name="triplet_accuracy",
            xlabel="Batch",
            train_keys=["batch/triplet_accuracy", "batch/train/triplet_accuracy"],
            val_keys=None,
        )

        # Plot 2: Top-right - Epoch Triplet Accuracy
        self._plot_subplot(
            axes[0, 1],
            title_prefix="Epoch",
            base_metric_name="triplet_accuracy",
            xlabel="Epoch",
            train_keys=["train/triplet_accuracy", "triplet_accuracy"],
            val_keys=["validation/triplet_accuracy", "triplet_accuracy"],
        )

        # Plot 3: Bottom-left - Batch Loss
        self._plot_subplot(
            axes[1, 0],
            title_prefix="Batch",
            base_metric_name="loss",
            xlabel="Batch",
            train_keys=["batch/loss", "batch/train/loss"],
            val_keys=None,
        )

        # Plot 4: Bottom-right - Epoch Loss
        self._plot_subplot(
            axes[1, 1],
            title_prefix="Epoch",
            base_metric_name="loss",
            xlabel="Epoch",
            train_keys=["train/loss", "loss"],
            val_keys=["validation/loss", "loss"],
        )

        os.makedirs(self.log_dir, exist_ok=True)

        plt.tight_layout(pad=2.0)
        fig.savefig(os.path.join(self.log_dir, "metrics_plot.png"), dpi=150)
        plt.show()
        plt.close(fig)
