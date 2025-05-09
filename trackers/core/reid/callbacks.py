from typing import Optional, Any
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[Any] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        self.writer = SummaryWriter(log_dir)

    def on_train_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_validation_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)
    
    def on_end(self):
        self.writer.flush()
        self.writer.close()
