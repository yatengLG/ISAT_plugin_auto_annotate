# -*- coding: utf-8 -*-
# @Author  : LG

import cv2
import numpy as np
import onnxruntime as ort


class DetectModel:
    def __init__(self, model, score_threshold=0.25):
        self.session = ort.InferenceSession(model)
        self.score_threshold = score_threshold
        _, _, self.h, self.w = self.session.get_inputs()[0].shape

    def preprocess(self, img: np.ndarray) -> (np.ndarray, float):

        shape = img.shape[:2]

        r = min(self.h / shape[0], self.w / shape[1])
        r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.h - new_unpad[0], self.w - new_unpad[1]

        top, bottom = 0, int(round(dh + 0.1))
        left, right = 0, int(round(dw + 0.1))

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = np.array(img).transpose((2, 0, 1))[None] / 255.0
        return img, r

    def __call__(self, img: str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img.shape[:2]

        img, r = self.preprocess(img)

        outputs = self.session.run(['output0'], {'images': img.astype(np.float32)})
        outputs = outputs[0][0]
        outputs = outputs[outputs[:, 4] > self.score_threshold]

        #
        outputs = outputs / np.array([r, r, r, r, 1, 1])
        outputs[:, 0] = np.clip(outputs[:, 0], 0, ori_w)
        outputs[:, 1] = np.clip(outputs[:, 1], 0, ori_h)
        outputs[:, 2] = np.clip(outputs[:, 2], 0, ori_w)
        outputs[:, 3] = np.clip(outputs[:, 3], 0, ori_h)

        return outputs



