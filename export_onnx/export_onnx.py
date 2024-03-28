import argparse
import cv2
import os
import time
import numpy as np
import torch
import imageio
from PIL import Image

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection
from config import build_dataset_config, build_model_config
from models import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # class label config
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False, 
                        help='show 14 action pose of AVA.')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    return parser.parse_args()

if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    x = torch.randn((1, 3, args.len_clip, args.img_size, args.img_size))

    output_names = ['conf_preds0', 'conf_preds1', 'conf_preds2', 'cls_preds0', 'cls_preds1', 'cls_preds2', 'reg_preds0', 'reg_preds1', 'erg_preds2']
    onnxpath = os.path.join('weights', os.path.splitext(os.path.basename(args.weight))[0]+'.onnx')
    with torch.no_grad():
        torch.onnx.export(model, x, onnxpath, 
                          input_names=["input"],
                          output_names=output_names, opset_version=17)   ###你也可以修改opset_version的版本, opset_version=17时onnxruntime版本需要1.12以上
    print('export', onnxpath, 'finish')


###python3 export_onnx.py -d ucf24 -v yowo_v2_nano -size 224 --weight pth_weights/yowo_v2_nano_ucf24.pth
###python3 export_onnx.py -d ucf24 -v yowo_v2_nano -size 224 --weight pth_weights/yowo_v2_nano_ucf24_k32.pth --len_clip=32
###python3 export_onnx.py -d ava_v2.2 -v yowo_v2_nano -size 224 --weight pth_weights/yowo_v2_nano_ava.pth
###python3 export_onnx.py -d ava_v2.2 -v yowo_v2_nano -size 224 --weight pth_weights/yowo_v2_nano_ava_k32.pth --len_clip=32
