import gc, torch
from detectron2.engine import DefaultTrainer, HookBase

import sample_config


class CustomHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()


def run(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([CustomHook()])
    trainer.train()


if __name__ == '__main__':
    _cfg = sample_config.run()
    run(_cfg)
