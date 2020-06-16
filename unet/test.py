from model import unet
from data import testGenerator
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import cv2

if __name__ == "__main__":

    testGene = testGenerator("../dataset/membrane/test/image", 5)
    model = unet()
    model.load_weights("unet_membrane.hdf5")
    results = model.predict_generator(testGene, 5, verbose=1)

    # saveResult("data/membrane/test",results)
    # print(type(results))
    # print(results)
    mask_pos = results >= 0.5
    mask_neg = results < 0.5
    results[mask_pos] = 1
    results[mask_neg] = 0

    print(mask_pos.sum())
    print(mask_neg.sum())
    for i in range(results.shape[0]):

        print(results[i].shape)
        print(results[i].squeeze().shape)
        array = np.uint8(results[i].squeeze())
        # print(array[:100, 0])
        array = array * 255
        img = Image.fromarray(array, 'L')
        img.save('{}.png'.format(i))
        # cv2.imwrite('{}.png'.format(i), array)
