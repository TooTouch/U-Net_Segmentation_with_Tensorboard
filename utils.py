import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import io

from keras.callbacks import TensorBoard

class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, log_dir, histogram_freq, **kwargs):
        super(TensorBoardWrapper, self).__init__(log_dir, histogram_freq, **kwargs)

        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps # Number of times to call next() on the generator.
        self.histogram_freq = histogram_freq

    def set_model(self, model):
        return super(TensorBoardWrapper, self).set_model(model)

    def make_image(self, tensor, convert=False):
        height, width, channel = tensor.shape
        if convert:
            tensor = (tensor * 255)
        tensor = tensor.astype(np.uint8)
        image = Image.fromarray(tensor.reshape(tensor.shape[:-1]))
        output = io.BytesIO()
        
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()

        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            image_batch, label_batch = next(self.batch_gen)
            if imgs is None and tags is None:

                imgs = np.zeros((image_batch.shape), dtype=np.float32)
                tags = np.zeros((label_batch.shape), dtype=np.uint8)
                print('imgs shape: ',imgs.shape)
                print('tags shape: ',tags.shape)
                print('ib shape: ',image_batch.shape)
                print('tb shape: ',label_batch.shape)

            imgs[s * image_batch.shape[0]:(s + 1) * image_batch.shape[0]] = image_batch
            tags[s * label_batch.shape[0]:(s + 1) * label_batch.shape[0]] = label_batch
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]

        if epoch % self.histogram_freq == 0:
            val_data = self.validation_data
            # Load image
            valid_images = val_data[0]  # X_train
            valid_labels = val_data[1]  # Y_train
            pred_images = self.model.predict(valid_images)

            summary_str = list()
            for i in tqdm(range(len(pred_images))):
                valid_image = self.make_image(valid_images[i], convert=True)
                valid_label = self.make_image(valid_labels[i], convert=True)
                pred_image = self.make_image(pred_images[i], convert=True)

                summary_str.append(tf.Summary.Value(tag='plot/%d/image' % i, image=valid_image))
                summary_str.append(tf.Summary.Value(tag='plot/%d/label' % i, image=valid_label))
                summary_str.append(tf.Summary.Value(tag='plot/%d/pred' % i, image=pred_image))


            self.writer.add_summary(tf.Summary(value = summary_str), epoch)

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)