import cv2, torch
import numpy as np
import pprint as pp
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from components.rav_ROI_heads import RAVROIHeads
from components.counter_ROI_heads import CounterROIHeads

input_path = './output/'
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# compare inference result of two model by Visualizer
class Model:
    def __init__(self, model_name):
        self.cfg = get_cfg()

        # get model and create predictor
        if model_name is None:
            # get baseline from Detectron2 model zoo
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        else:
            self.cfg.merge_from_file(input_path + model_name + '/output.yaml')
            self.cfg.MODEL.WEIGHTS = input_path + model_name + '/model_final.pth'

        self.predictor = DefaultPredictor(self.cfg)

    # draw prediction result on input image
    def draw(self, im):
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_img = out.get_image()[:, :, ::-1]

        # organize scores for each instance
        result = {}
        for idx, label in enumerate(outputs['instances'].pred_classes):
            result[class_names[label]+str(idx)] = outputs['instances'].scores[idx].item()

        return out_img, result

    # calculate the number of total trainable parameters in model
    def parameters(self):
        model = self.predictor.model

        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param

        return total_params


# make inference for image in './sample/' directory and visualize
# each image file has name input + (index) with jpg format, index starts from 0
# left side shows inference results of model1_name
def run(model1_name, model2_name=None):
    cmp = Model(model1_name)
    base = Model(model2_name)

    i = 0
    while True:
        i = i + 1
        try:
            im = cv2.imread("./sample/input" + str(i) + ".jpg")

            ic, rc = cmp.draw(im)
            ib, rb = base.draw(im)

            # print score for each object
            pp.pprint(rc)
            pp.pprint(rb)

        except AttributeError:
            # no more images
            break

        ic_r = ResizeWithAspectRatio(ic, width=900)
        ib_r = ResizeWithAspectRatio(ib, width=900)
        cv2.imshow('result', np.concatenate((ic_r, ib_r), axis=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# resize image
# source: https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# inference example
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run('rav_net_64', 'baseline_counter')
