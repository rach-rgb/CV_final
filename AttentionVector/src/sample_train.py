import torch
from detectron2.config import get_cfg

from val_hook import GCHook
from relevance_trainer import RelTrainer
from components.cross_ROI_heads import CrossROIHeads


def run():
    cfg = get_cfg()
    cfg.merge_from_file('output.yaml')

    # train
    trainer = RelTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([GCHook()])
    trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run()

