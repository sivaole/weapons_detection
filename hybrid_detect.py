import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2 import model_zoo
from time import time
import xml.etree.ElementTree as ET
import PIL.Image as Image
import json
import urllib
from tqdm import tqdm
import pandas as pd
import pickle
import itertools
import random
import cv2
import numpy as np
import ntpath
import glob
import detectron2
import torchvision
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.init()
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00",
                        "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv('../weapons_anns.csv')
classes = df.class_name.unique().tolist()

train_df = pd.read_csv('../df_train.csv')
test_df = pd.read_csv('../df_val.csv')

IMAGES_PATH = f'../weapons_data'


def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):

        record = {}

        image_df = df[df.file_name == img_name]

        file_path = f'{IMAGES_PATH}/images/train/{img_name}'
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)

        objs = []
        for _, row in image_df.iterrows():

            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)

            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(row.class_name),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# Register dataset and metadata catologues
DatasetCatalog._REGISTERED.clear()
# del DatasetCatalog._REGISTERED['weapons_train']
# del DatasetCatalog._REGISTERED['weapons_val']
print(DatasetCatalog._REGISTERED)
print(len(DatasetCatalog._REGISTERED))

entriesToRemove = ('weapons_train', 'weapons_val')
for k in entriesToRemove:
    DatasetCatalog._REGISTERED.pop(k, None)

for d in ["train", "val"]:
    DatasetCatalog.register("weapons_" + d, lambda d=d: create_dataset_dicts(
        train_df if d == "train" else test_df, classes))
    MetadataCatalog.get("weapons_" + d).set(thing_classes=classes)

statement_metadata = MetadataCatalog.get("weapons_train")


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
            # output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()

# The evaluation results will be stored in the `coco_eval` folder if no folder is provided.

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)

cfg.DATASETS.TRAIN = ("weapons_train",)
cfg.DATASETS.TEST = ("weapons_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

cfg.TEST.EVAL_PERIOD = 500
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join("../output/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

predictor = DefaultPredictor(cfg)


evaluator = COCOEvaluator("weapons_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "weapons_val")
inference_on_dataset(trainer.model, val_loader, evaluator)


# experiment_folder = './output/model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'
experiment_folder = './output/'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_metrics = load_json_arr(experiment_folder + 'metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics],
    [x['total_loss'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()

# os.makedirs("weapons_results", exist_ok=True)

test_image_paths = test_df.file_name.unique()


def detect(save_img=False):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(
        device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split(
            '.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()
              ) if device.type != 'cpu' else None  # run once

    times = []
    info = {}

    for path, img, im0s, vid_cap in dataset:
        start_time = time()
        fname = path.split('/')[-1]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Pass on to Detectron if Classes detected
        if pred != [None]:
            im = cv2.imread(path)
            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=statement_metadata,
                scale=1.,
                instance_mode=ColorMode.IMAGE
            )
            instances = outputs["instances"].to("cpu")
            instances.remove('pred_masks')
            end_time = time()
            info[fname] = [str(instances), end_time-start_time]

            v = v.draw_instance_predictions(instances)
            result = v.get_image()[:, :, ::-1]
            file_name = ntpath.basename(path)
            write_res = cv2.imwrite(f'output_hybrid/{file_name}', result)
        else:
            im = cv2.imread(path)
            file_name = ntpath.basename(path)

            end_time = time()
            info[fname] = ['', end_time-start_time]

            write_res = cv2.imwrite(f'output_hybrid/{file_name}', im)
        # print(pred)
        # print(outputs)

        # times.append(str(end_time-start_time))
    print('Done. (%.3fs)' % (time() - t0))

    with open('output_hybrid/detection_out.pckl', 'wb') as f:
        pickle.dump(info, f)

    with open('output_hybrid/times.txt', 'w') as t:
        t.write('\n'.join(times))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str,
                        default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str,
                        default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/samples', help='source')
    parser.add_argument('--output', type=str, default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true',
                        help='half precision FP16 inference')
    parser.add_argument('--device', default='',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+',
                        type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
