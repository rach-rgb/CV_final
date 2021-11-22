import gc, torch
from detectron2.engine import DefaultTrainer, HookBase

import sample_config


class GCHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()


# Train only ROI Head
class ROIHeadTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.proposal_generator.parameters():
            param.requires_grad = False
        return model


def run(cfg):
    trainer = ROIHeadTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([GCHook()])
    trainer.train()


if __name__ == '__main__':
    _cfg = sample_config.run()
    run(_cfg)
