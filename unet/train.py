from model import unet
from data import trainGenerator
from keras.callbacks import ModelCheckpoint
import argparse
import os


def train(args):
    # data augmentation configuration
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    # training set
    myGene = trainGenerator(2, os.path.join(args.root, 'train'),
                            'image', 'label', data_gen_args, target_size=(512, 512), save_to_dir=None)
    # model 
    model = unet()
    model_checkpoint = ModelCheckpoint(
        args.save_path, monitor='loss', verbose=1, save_best_only=True)
    # train
    model.fit_generator(myGene, steps_per_epoch=2000,
                        epochs=5, callbacks=[model_checkpoint])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../dataset/membrane')
    parser.add_argument('--save_path', type=str, default='unet_membrane.hdf5')
    args = parser.parse_args()

    train(args)