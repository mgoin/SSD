import glob
import os
import sys
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

import cv2
import time

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer

from imutils.video import WebcamVideoStream
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


class FPS():
    def __init__(self, avg=10):
        self.fps = np.empty(avg)
        self.t0 = time.clock()

    def tick(self):
        t = time.clock()
        self.fps[1:] = self.fps[:-1]
        self.fps[0] = 1./(t-self.t0)
        self.t0 = t
        return self.fps.mean()


fps = FPS(100)


class App(QtGui.QMainWindow):
    @torch.no_grad()
    def __init__(self, cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
        super(App, self).__init__()

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        # start video stream thread, allow buffer to fill
        print("[INFO] starting threaded video stream...")
        self.stream = WebcamVideoStream(src=0).start()  # default camera
        time.sleep(0.2)
        test_image = self.stream.read()

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(
            0, 0, test_image.shape[0], test_image.shape[1]))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        if dataset_type == "voc":
            class_names = VOCDataset.class_names
        elif dataset_type == 'coco':
            class_names = COCODataset.class_names
        else:
            raise NotImplementedError('Not implemented now.')
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.model = build_detection_model(cfg)
        self.model = self.model.to(self.device)
        checkpointer = CheckPointer(self.model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(ckpt, use_latest=ckpt is None)
        weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
        print('Loaded weights from {}'.format(weight_file))

        self.cpu_device = torch.device("cpu")
        self.transforms = build_transforms(cfg, is_train=False)
        self.model.eval()

        if dataset_type == "voc":
            self.class_names = VOCDataset.class_names
        elif dataset_type == 'coco':
            self.class_names = COCODataset.class_names
        else:
            raise NotImplementedError('Not implemented now.')
        self.score_threshold = score_threshold

        #### Start  #####################
        self._update()

    @torch.no_grad()
    def _update(self):
        # grab next frame
        start = time.time()
        image = np.array(self.stream.read())
        height, width = image.shape[:2]
        images = self.transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = self.model(images.to(self.device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(self.cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > self.score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        drawn_image = draw_boxes(image, boxes, labels,
                                 scores, self.class_names).astype(np.uint8)

        # keybindings for display
        # if key == ord('p'):  # pause
        #     while True:
        #         key2 = cv2.waitKey(1) or 0xff
        #         cv2.imshow('image', drawn_image)
        #         if key2 == ord('p'):  # resume
        #             break
        # cv2.imshow('image', drawn_image)
        # if key == 27:  # exit
        #     break

        start = time.time()
        self.img.setImage(drawn_image)
        tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps=fps.tick())
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        gui_time = time.time() - start

        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'gui {:02d}ms'.format(round(gui_time * 1000))
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print(meters)


def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(0.5)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        start = time.time()
        image = np.array(stream.read())
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        key = cv2.waitKey(1) & 0xFF

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print(meters)

        drawn_image = draw_boxes(image, boxes, labels,
                                 scores, class_names).astype(np.uint8)
        # Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('image', drawn_image)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('image', drawn_image)
        if key == 27:  # exit
            break


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo',
                        type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result',
                        type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    # run_demo(cfg=cfg,
    #          ckpt=args.ckpt,
    #          score_threshold=args.score_threshold,
    #          images_dir=args.images_dir,
    #          output_dir=args.output_dir,
    #          dataset_type=args.dataset_type)
    app = QtGui.QApplication(sys.argv)
    thisapp = App(cfg=cfg,
                  ckpt=args.ckpt,
                  score_threshold=args.score_threshold,
                  images_dir=args.images_dir,
                  output_dir=args.output_dir,
                  dataset_type=args.dataset_type)
    thisapp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
