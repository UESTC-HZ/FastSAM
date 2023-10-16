import argparse
import os

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

from fastsam import FastSAM, FastSAMPrompt
from label_process import segment_boxes, remove_small_regions, remove_small_block


def seg_image_process(data_path, save_path, model_type, edge):
    model = FastSAM('./models/' + model_type)
    # DEVICE = '[0,1,2,3]'
    DEVICE = '0'

    data_path = os.path.join(data_path)
    class_list = os.listdir(data_path)

    index = 0
    for cls in class_list:
        print('The number of completed categories :' + str(index))
        index += 1
        image_path = os.path.join(data_path, cls)
        new_image_path = os.path.join(save_path, cls)

        if not os.path.exists(new_image_path):
            os.makedirs(new_image_path)

        for name in tqdm(os.listdir(image_path), desc="calss : " + str(index - 1)):
            img_path = os.path.join(image_path, name)
            image = Image.open(img_path)  # 加载图片
            if image.mode != "RGB":
                image = image.convert("RGB")
                # image.save(os.path.join(new_image_path, name))
                # continue

            image = np.asarray(image)
            h, w, c = image.shape
            new_mask = np.zeros((h, w))

            hh = h * edge
            ww = w * edge

            #  图像分块策略
            # boxes = segment_boxes(h, w, edge)

            boxes = [[ww, hh, w - ww, h - hh], [ww, 0, w - ww, h - hh], [ww, hh, w - ww, h], [0, hh, w - ww, h - hh],
                     [ww, hh, w, h - hh]]

            everything_results = model(img_path, device=DEVICE, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, )
            prompt_process = FastSAMPrompt(img_path, everything_results, device=DEVICE)
            masks = prompt_process.box_prompt(bboxes=boxes)

            if len(masks) == 0:
                new_mask.fill(1)
            else:
                for mask in masks:
                    if mask.shape == new_mask.shape:
                        new_mask += mask

            new_mask = new_mask.astype(np.uint8)
            rgb_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2RGB)
            # new_image = np.where(rgb_mask > 0, 0, image)
            new_image = np.where(rgb_mask > 0, image, 0)

            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(new_image_path, name), new_image)


def args_parser():
    parser = argparse.ArgumentParser()
    # /data/ImageNet100/train/
    parser.add_argument('data_path', type=str, help="Path of the original dataset.")
    parser.add_argument('save_path', type=str, help="Path of the new dataset.")
    parser.add_argument('--model_type', type=str, default='FastSAM-x.pt', help="FastSAM-s.pt,FastSAM-x.pt")
    parser.add_argument('--edge', type=float, default=0.3, help="The distance for edge cropping")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    data_path = args.data_path
    save_path = args.save_path
    model_type = args.model_type
    edge = args.edge
    if not os.path.exists(data_path):
        print("Dataset not found in " + data_path)
        return
    seg_image_process(data_path, save_path, model_type, edge)


if __name__ == '__main__':
    args = args_parser()
    main(args)
