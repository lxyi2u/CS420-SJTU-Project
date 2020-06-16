from model import unet
from data import trainGenerator
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(2, '../dataset/membrane/train',
                            'image', 'label', data_gen_args, save_to_dir=None)
    model = unet()
    model_checkpoint = ModelCheckpoint(
        'unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=2000,
                        epochs=5, callbacks=[model_checkpoint])
