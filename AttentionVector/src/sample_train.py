import torch
from detectron2.config import get_cfg

from components.rav_ROI_heads import RAVROIHeads
from components.counter_ROI_heads import CounterROIHeads
from relevance_trainer import RelevanceTrainer, TuningTrainer


# train example
def run(rel_train, resume, remove_all):
    cfg = get_cfg()
    cfg.merge_from_file('output.yaml')

    # train
    if rel_train:
        trainer = RelevanceTrainer(cfg)
        trainer.resume_or_load(resume, remove_all)
        trainer.train()
    else:
        trainer = TuningTrainer(cfg)
        trainer.resume_or_load(resume, remove_all)
        trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run(rel_train=True, resume=False, remove_all=False)

