from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, List, Optional


class WandbVideoLogger(WandbLogger):
    @property
    def name(self):
        return "Wandb Logger with Video Support"
    
    @rank_zero_only
    def log_video(self, key: str, videos: List[Any], step: Optional[int]=None):
        """Log a video to wandb.

        Args:
            key (str): Key to log the videos to in Wandb 
            video (list): List of Wandb Video Objects to log to wandb
            step (int): The global step for this video. Defaults to the current global step.
        """
        self.log_metrics({key: videos}, step=step)