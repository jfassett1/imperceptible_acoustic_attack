from pytorch_lightning.callbacks import Callback
import torch
from typing import Literal
import torch.distributed as dist
from pytorch_lightning.callbacks.progress import TQDMProgressBar


# For displaying current predicted etoken
class TokenDisplayProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.decoded_token_str = ""

    def set_token(self, token_str: str):
        self.decoded_token_str = token_str

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics["pred"] = self.decoded_token_str  # Add decoded string to progress bar
        return metrics

class FinetuningCallback(Callback):
    def __init__(self, patience_steps=50, check_interval=3):
        self.best_val_loss = float('inf')
        self.bad_steps = 0
        self.patience_steps = patience_steps
        self.check_interval = check_interval
        self.step_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        val_loss = outputs
        self.step_count += 1

        if self.step_count % self.check_interval == 0:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.bad_steps = 0
            else:
                self.bad_steps += self.check_interval

            if self.bad_steps >= self.patience_steps and not pl_module.stage == 2:
                pl_module.stage = 2
                print(f"Finetuning activated at step {self.step_count}")

class LossThresholdCallback(Callback):
    def __init__(self, threshold, metric = 'loss'):
        self.threshold = threshold
        self.metric = metric

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs[self.metric] if isinstance(outputs, dict) else outputs.item()

        # Broadcast stop condition from rank 0
        should_stop = torch.tensor([loss < self.threshold], device=pl_module.device)
        
        if trainer.world_size > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)  # Any rank stopping will trigger all to stop

        if should_stop.item():
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        # Ensure all ranks wait here before moving on
        if trainer.world_size > 1:
            trainer.strategy.barrier()

class ValLossCallback(Callback):
    def __init__(self, threshold: float,metric:str = "val_per",comp=Literal['lesser',"greater"]):
        super().__init__()
        self.threshold = threshold
        self.metric = metric
        self.comp = comp
        
    def on_validation_end(self, trainer, pl_module):
        should_stop = torch.tensor(0, device=pl_module.device)
        val_loss = trainer.callback_metrics.get(self.metric)
        if self.comp == "greater":
            if val_loss is not None and val_loss > self.threshold:
                print(f"Stopping training: {self.metric} {val_loss:.4f} > threshold {self.threshold}")
                should_stop = torch.tensor(1, device=pl_module.device)
        elif self.comp == "lesser":
            if val_loss is not None and val_loss < self.threshold:
                print(f"Stopping training: {self.metric} {val_loss:.4f} < threshold {self.threshold}")
                should_stop = torch.tensor(1, device=pl_module.device)
        else:
            raise KeyError
        # Send signal to all GPUs
        if trainer.world_size > 1 and dist.is_initialized():
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)   

        if should_stop.item() == 1:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        # Ensure all ranks wait here before moving on
        if trainer.world_size > 1:
            trainer.strategy.barrier()

class LrValActivate(Callback): 
    """
    Callback for activating lr settings.
    Aims for maximal validation prediction power.
    Lowers Learning rate, and increases validation checking interval
    
    """
    def __init__(self, threshold:float):
        super().__init__()
        self.threshold = threshold
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("loss_step")
        # If train loss reaches certain threshold, increase the validation checking. Also make a new threshold lower
        if loss < self.threshold:
            print("Lowering Learning Rate")
            trainer.val_check_interval = trainer.val_check_interval / 2
            self.threshold = self.threshold / 2
            optimizer = pl_module.trainer.optimizers[0]
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    