from model import *
from data import *
from utils import *
import keras.callbacks as kb
import os


def main():
    # directory
    history_dir = os.path.join(os.getcwd(),'history')
    log_dir = os.path.join(os.getcwd(),'logs')

    # create directiry
    if not os.path.isdir(history_dir):
        os.mkdir(history_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # setting
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    nb_epochs = 10
    batch_size = 5
    train_length = 25
    valid_length = 5
    nb_aug = 10
    valid_steps = int(valid_length/batch_size)
    log_id = len(os.listdir('logs'))

    # generator parameters
    data_gen_args = dict(rescale=1./255,
                         rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='reflect')

    # generator
    myGene = trainGenerator(batch_size, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir = None)
    myGene_valid = validGenerator(batch_size, 'data/membrane/validation', 'image', 'label')

    # model
    model = unet()

    # callbacks
    model_checkpoint = kb.ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    tb = TensorBoardWrapper(myGene_valid, nb_steps=valid_steps, log_dir='./logs/%d'%log_id, histogram_freq=1, batch_size=1, write_graph=True, write_grads=True, write_images=False)
    csv_logger = kb.CSVLogger('./history/%d.csv'%log_id, append=True)

    callbacks = [model_checkpoint, tb, csv_logger]

    # train
    model.fit_generator(myGene,
                        steps_per_epoch=(train_length/batch_size) * nb_aug,
                        epochs=nb_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=myGene_valid,
                        validation_steps=valid_length/batch_size)

    # test
    testGene = testGenerator("data/membrane/test")
    results = model.predict_generator(testGene,30,verbose=1)
    saveResult("data/membrane/test",results)


if __name__=='__main__':
    main()
