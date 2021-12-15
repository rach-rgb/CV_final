import os, copy
import pandas as pd
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
setup_logger()

from rav_ROI_heads import RAVROIHeads
from components.counter_ROI_heads import CounterROIHeads

metrics = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-person', 'AP-bicycle', 'AP-car', 'AP-motorcycle',
           'AP-airplane', 'AP-bus', 'AP-train', 'AP-truck', 'AP-boat', 'AP-traffic light', 'AP-fire hydrant',
           'AP-stop sign', 'AP-parking meter', 'AP-bench', 'AP-bird', 'AP-cat', 'AP-dog', 'AP-horse', 'AP-sheep',
           'AP-cow', 'AP-elephant', 'AP-bear', 'AP-zebra', 'AP-giraffe', 'AP-backpack', 'AP-umbrella', 'AP-handbag',
           'AP-tie', 'AP-suitcase', 'AP-frisbee', 'AP-skis', 'AP-snowboard', 'AP-sports ball', 'AP-kite',
           'AP-baseball bat', 'AP-baseball glove', 'AP-skateboard', 'AP-surfboard', 'AP-tennis racket', 'AP-bottle',
           'AP-wine glass', 'AP-cup', 'AP-fork', 'AP-knife', 'AP-spoon', 'AP-bowl', 'AP-banana', 'AP-apple',
           'AP-sandwich', 'AP-orange', 'AP-broccoli', 'AP-carrot', 'AP-hot dog', 'AP-pizza', 'AP-donut',
           'AP-cake', 'AP-chair', 'AP-couch', 'AP-potted plant', 'AP-bed', 'AP-dining table', 'AP-toilet', 'AP-tv',
           'AP-laptop', 'AP-mouse', 'AP-remote', 'AP-keyboard', 'AP-cell phone', 'AP-microwave', 'AP-oven',
           'AP-toaster', 'AP-sink', 'AP-refrigerator', 'AP-book', 'AP-clock', 'AP-vase', 'AP-scissors',
           'AP-teddy bear', 'AP-hair drier', 'AP-toothbrush']


# evaluate bbox AP for metrics
def run(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(input_path + model_name + '/output.yaml')
    cfg.MODEL.WEIGHTS = input_path + model_name + '/model_final.pth'

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir="output")
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")

    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    result_dict[model_name+' soft'] = copy.deepcopy(result['bbox'])


if __name__ == "__main__":
    os.chdir('../')
    input_path = './output/'
    result_path = './output/AP.csv'

    result_dict = {}
    result_df = pd.read_csv(result_path, index_col=0).transpose()
    result_dict.update(result_df.to_dict())

    run('rav_net_64')

    # save AP as dataframe
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=metrics)

    result_df.to_csv(result_path)
    print(result_df[['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']])

