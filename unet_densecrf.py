#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/6/19 14:04
#@Author: Shuyuan Fu
#@File  : unet_denseCRF.py
# This file is used to perform denseCRF on the score map
# obtained by Unet.

import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
from model import unet
from data import testGenerator, labelGenerator
from PIL import Image
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


def dense_crf(i,img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]
    print("output_probs.shape", output_probs.shape)

    # get unary
    d = dcrf.DenseCRF2D(w, h, 2)
    U = np.expand_dims(-np.log(output_probs), axis=0) # [1,H,W]
    U_ = np.expand_dims(-np.log(1-output_probs), axis=0) # [1,H,W]
    unary = np.concatenate((U_, U), axis=0)
    unary = unary.reshape((2, -1)) # [2,HW]
    d.setUnaryEnergy(unary)  # Unary
    print("U", U)

    # binary potential
    '''
        addPairwiseGaussian函数里的sxy为公式中的 $\theta_{\gamma}$, 
        addPairwiseBilateral函数里的sxy、srgb为$\theta_{\alpha}$ 和 $\theta_{\beta}$
    '''
    d.addPairwiseGaussian(sxy=1, compat=10)
    d.addPairwiseBilateral(sxy=5, srgb=0.01, rgbim=img, compat=10)

    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w)).astype(np.float32)
    print("Q", Q)
    return Q

def unet_crf():
    model = unet()
    model.load_weights("./unet_membrane_512_1e-5.hdf5")
    testGene = testGenerator("./dataset/membrane/test/img", 1)
    labelGene = labelGenerator("./dataset/membrane/test/label", 1)
    dataset = zip(testGene, labelGene)
    for idx, (data, target) in enumerate(dataset):
        pred = model.predict(data, 1, verbose=1)
        pred = pred.squeeze()
        print("pred",pred.shape)

        # original image, 0-255, 3 channel
        img = cv2.imread("./dataset/membrane/test/img/%s" % str(idx)+ ".png")
        # perform dense crf
        final_mask = dense_crf(idx, np.array(img).astype(np.uint8), pred)
        # binarization score map to get predict result of Unet
        mask_pos = pred >= 0.5
        mask_neg = pred < 0.5
        pred[mask_pos] = 1
        pred[mask_neg] = 0
        pred = np.uint8(pred)
        pred = pred * 255
        pred = Image.fromarray(pred, 'L')
        pred.save('./{}.png'.format(idx))
        # draw result after denseCRF
        final_mask = np.uint8(final_mask)
        final_mask = final_mask * 255
        crf = Image.fromarray(final_mask, 'L')
        crf.save('./img/{}.png'.format(idx))


if __name__ == "__main__":
    unet_crf()