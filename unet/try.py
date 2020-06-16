from model import unet
from data import testGenerator
from keras.callbacks import ModelCheckpoint


if __name__ == "__main__":

    testGene = testGenerator("../dataset/membrane/test", 5)
    model = unet()
    model.load_weights("unet_membrane.hdf5")
    results = model.predict_generator(testGene, 5, verbose=1)

    # saveResult("data/membrane/test",results)
    print(results.shape)
