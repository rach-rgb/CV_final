import os
from torch import nn
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader

from validate_hook import LossEvalHook
from relevance_loader import get_rel_classes


# Trainer for fine tune
class TuningTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # freeze backbone and proposal net
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.proposal_generator.parameters():
            param.requires_grad = False

        return model

    # source: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py-L14
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        del hooks[4]  # delete predefined loss hook
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            ),
            50
        ))

        return hooks

    def resume_or_load(self, resume=True, remove_all=False):
        super().resume_or_load(resume)
        if not resume:  # initialize cls_score again if weight are loaded from cfg
            # cls_score
            nn.init.normal_(self.model.roi_heads.box_predictor.cls_score.weight, std=0.01)

            if remove_all:  # initialize box_head and bbox_pred
                # box_head
                nn.init.normal_(self.model.roi_heads.box_head.fc1.weight, std=0.01)
                nn.init.normal_(self.model.roi_heads.box_head.fc2.weight, std=0.01)
                # bbox_pred
                nn.init.normal_(self.model.roi_heads.box_predictor.bbox_pred.weight, std=0.01)


# Custom Trainer
class RelevanceTrainer(TuningTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # build relevance matrix from train data
        if model.roi_heads.box_predictor.cross_net.prior_rel[0][1] == 0:
            rel = get_rel_classes(cfg)
            model.roi_heads.box_predictor.cross_net.set_rel(rel)

        # freeze backbone and proposal net
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.proposal_generator.parameters():
            param.requires_grad = False

        return model

