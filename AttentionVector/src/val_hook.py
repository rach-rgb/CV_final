import gc, torch
from detectron2.engine import DefaultTrainer, HookBase


# Cuda Garbage Collection
class GCHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()