from detectron2.engine import DefaultTrainer

from rel_data_loader import get_rel_classes


# Custom Trainer
class PartialTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # build relevance matrix from train data
        rel = get_rel_classes(cfg)
        model.roi_heads.box_predictor.cross_net.set_rel(rel)

        # freeze backbone and proposal net
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.proposal_generator.parameters():
            param.requires_grad = False
        return model
