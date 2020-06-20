from model import unet
from data import trainGenerator, testGenerator, labelGenerator
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import cv2
import argparse
import os



def draw(model_path, image_path):

    model = unet()
    model.load_weights(model_path)
    testGene = testGenerator(image_path, 5, target_size=(512,512))
    results = model.predict_generator(testGene, 5, verbose=1)
    results = results > 0.5

    for i in range(results.shape[0]):
        array = np.uint8(results[i].squeeze())
        array = array * 255
        img = Image.fromarray(array, 'L')
        img.save('{}.png'.format(i))

def cal_acc(model_path, image_path, label_path, num_image):
    model = unet()
    model.load_weights(model_path)
    testGene = testGenerator(image_path, num_image, target_size=(512, 512))
    labelGene = labelGenerator(label_path, num_image, target_size=(512, 512))
    dataset = zip(testGene, labelGene)
    acc = 0
    total_num = 0
    for idx, (data,target) in enumerate(dataset):

        pred = model.predict(data, 1, verbose=1)
        pred = pred > 0.5
        pred = pred.squeeze()
        acc += (target == pred).mean()
        total_num += data.shape[0]

    print('Avg Accuracy:', acc / total_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../dataset/membrane')
    parser.add_argument('--model_path', type=str, default='unet_membrane_512_1e-5_aug.hdf5')
    args = parser.parse_args()

    # calculate test accuracy
    cal_acc(model_path=args.model_path,
            image_path=os.path.join(args.root, "test/image"),
            label_path=os.path.join(args.root, "test/label"),
            num_image=5)
    # calculate train accuracy
    cal_acc(model_path=args.model_path,
            image_path=os.path.join(args.root, "train/image"),
            label_path=os.path.join(args.root, "train/label"),
            num_image=20)
    # draw the prediction image
    draw(model_path=args.model_path, 
            image_path=os.path.join(args.root,"test/image"))