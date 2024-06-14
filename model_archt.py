import tensorflow as tf
import os


class SaveAndResumeCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_instance, checkpoint_dir):
        super(SaveAndResumeCallback, self).__init__()
        self.model_instance = model_instance
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "check_point.ckpt")

    def on_epoch_end(self, epoch, logs=None):
        # Save the model's weights at the end of each epoch
        self.model_instance.save_weights(
            self.checkpoint_path.format(epoch=epoch + 1))

    def on_train_begin(self, logs=None):
        # Check if there are existing checkpoints to resume training from
        if os.path.exists(self.checkpoint_dir):
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                print("Resuming training from checkpoint:", latest_checkpoint)
                self.model_instance.load_weights(latest_checkpoint)


class u_net_layer(tf.keras.Model):
    def __init__(self):
        super(u_net_layer, self).__init__()
        # self.inp = tf.keras.Input(
        #    shape=(input_size[0], input_size[1], input_size[2]))
        # first block
        """
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))"""
        # second block
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.conv4 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # third block
        self.conv5 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.conv6 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # 4th block
        self.conv7 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.conv8 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # 5th block
        self.conv9 = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.conv10 = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=(1, 1))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        # 6th block upsampling
        self.conv11 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(
            3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv12 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.upsamp1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # 7th block upsample
        self.conv13 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv14 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.upsamp2 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # 8th block upsample
        self.conv15 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv16 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.upsamp3 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # 9th block upsample
        self.conv17 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv18 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        # self.upsamp4 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # 10th block
        self.conv19 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01), kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv20 = tf.keras.layers.Conv2D(
            filters=2, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(), padding='same', strides=1)
        self.conv21 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same', strides=1)
        self.drop1 = tf.keras.layers.Dropout(0.5)
        # self.drop2 = tf.keras.layers.Dropout(0.5)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):

        self.batchnorm1.build(input_shape=(
            None, (input_shape[1]//4), (input_shape[1]//4), 256))
        self.batchnorm2.build(input_shape=(
            None, (input_shape[1]//2), (input_shape[1]//2), 256*2))

    @tf.function
    def call(self, inp, training=False):
        # x1 = self.inp(inp)
        # x1 = self.conv1(inp)  # 160
        # x1 = self.conv2(x1)
        # x2 = self.pool1(x1)

        x2 = self.conv3(inp)  # 80
        x2 = self.conv4(x2)
        x3 = self.pool2(x2)

        x3 = self.conv5(x3)  # 40
        x3 = self.conv6(x3)
        x4 = self.pool3(x3)

        x4 = self.batchnorm1(x4, training=training)  # 20
        x4 = self.conv7(x4)
        x4 = self.conv8(x4)
        x5 = self.pool4(x4)
        x5 = self.drop1(x5)

        x5 = self.conv9(x5)  # 10
        x5 = self.conv10(x5)

        x5 = self.conv11(x5)
        x5 = self.conv12(x5)
        x5 = self.upsamp1(x5)
        concat1 = tf.keras.layers.concatenate([x4, x5], axis=3)

        x6 = self.conv13(concat1)  # 20
        x6 = self.conv14(x6)
        x6 = self.upsamp2(x6)
        concat2 = tf.keras.layers.concatenate([x3, x6], axis=3)

        x7 = self.batchnorm2(concat2, training=training)  # 40
        x7 = self.conv15(x7)
        x7 = self.conv16(x7)
        x7 = self.upsamp3(x7)
        concat3 = tf.keras.layers.concatenate([x2, x7], axis=3)  # 80

        x8 = self.conv17(concat3)
        x8 = self.conv18(x8)
        # x8 = self.upsamp4(x8)
        # x8 = self.drop2(x8)
        # concat4 = tf.keras.layers.concatenate([x1, x8], axis=3)  # 160

        x9 = self.conv19(x8)  # (concat4)
        x9 = self.conv20(x9)
        x9 = self.conv21(x9)

        return x9


class u_net_model(tf.keras.Model):
    def __init__(self, is_train, inp_shape):
        super(u_net_model, self).__init__()
        self.inp_shape = inp_shape
        self.is_train = is_train
        self.unet = u_net_layer()
        self.unet.build(input_shape=(
            None, self.inp_shape[0], self.inp_shape[1], self.inp_shape[2]))

    def call(self, input_img, ylabel=None):

        output1 = self.unet(input_img, training=self.is_train)
        if self.is_train:
            loss = calculate_loss(ylabel, output1)
            return loss, output1
        else:
            return output1


def calculate_loss(y_true, y_pred):
    """use binary cross entrophy """
    bin_loss = tf.keras.losses.BinaryCrossentropy()

    loss = tf.reduce_sum(bin_loss(y_true, y_pred))
    return loss
