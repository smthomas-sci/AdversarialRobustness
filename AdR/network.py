"""

Builds a model with an fit() method that includes the option
to perform adversarial training.

NOTE:
 - At this stage it only works for ResNet50


Author: Simon Thomas
Date: 29-Sep-2020

"""

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

from AdR.utils import *
from AdR.data import *

# ---------- FIXES BIG --------------------------- #
tf.config.experimental_run_functions_eagerly(True)
# ------------------------------------------------ #


class AdversarialClassifier(tf.keras.models.Model):
    """
    The adversarial classifier model which can be trained
    directly using .fit() method.
    """
    def __init__(self, num_classes, input_shape=(224, 224, 3), weights="imagenet", **kwargs):
        """

        :param num_classes: the number of classes to classifier
        :param input_shape: the image shape. Default = (224, 224, 3)
        :param weights: path to weights. Default = "imagenet"
        """
        super(AdversarialClassifier, self).__init__(self)
        self.num_classes = num_classes
        self.dim = input_shape
        self.resNet = ResNet50(
                classes=num_classes,
                include_top=True,
                weights=weights,
                input_shape=input_shape
                )
        self.build(input_shape)

    def compile(self, classifier_optimizer, adversarial_optimizer, ε=3, adversarial_steps=7):
        """
        Overrides the compile step.
        """
        super(AdversarialClassifier, self).compile()
        # Used in training loop only
        self.optimizer_a = adversarial_optimizer
        self.ε = ε
        self.adversarial_steps = adversarial_steps
        self.clipper = lambda x: tf.clip_by_value(x, -self.ε, self.ε)
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Prepare resNet
        self.optimizer_c = classifier_optimizer
        self.resNet.get_layer("predictions").activation = None
        self.θ = self.resNet.trainable_variables

        # Create loss trackers
        self.a_loss_tracker = tf.keras.metrics.Mean(name="loss_a")
        self.c_loss_tracker = tf.keras.metrics.Mean(name="loss_c")
        self.acc_tracker = tf.keras.metrics.Mean(name="acc")


    def train_step(self, inputs):
        """
        Performs a single training step on a batch of inputs
        :param inputs: (x,y) of images and one-hot labels
        :return: losses
        """
        x, y_real = inputs

        batch_size = x.shape[0]
        if not batch_size:
            batch_size = 1

        # ------------------------- #
        #  Step I - Perturb inputs
        # ------------------------- #

        y_fake = tf.roll(y_real, 1, axis=-1)
        δ = tf.Variable(tf.zeros([batch_size] + list(self.dim),
                        dtype="float32"),
                        constraint=self.clipper,
                        #aggregation=tf.VariableAggregation.SUM,
                        #synchronization=tf.VariableSynchronization.AUTO
                        )
        # Perform n update steps of PGD
        for step in range(self.adversarial_steps):
            with tf.GradientTape() as tape:
                tape.watch(δ)
                # Add noise
                x_fake = tf.clip_by_value(x + δ, clip_value_min=-127, clip_value_max=127)
                y_pred = self.resNet(x_fake)

                # Compute loss
                loss_adv = (-self.cross_entropy(y_real, y_pred) +
                             self.cross_entropy(y_fake, y_pred) )
            # Calculate gradients and apply their signs
            grads = tape.gradient(loss_adv, [δ])
            self.optimizer_a.apply_gradients(zip([tf.math.sign(grads[0])], [δ]))

        # ---------------------------- #
        #  Step II - Update classifier
        # ---------------------------- #

        with tf.GradientTape() as tape:
            y_pred = self.resNet(x_fake)
            acc = tf.reduce_sum(tf.cast(tf.argmax(y_pred, axis=-1) == tf.argmax(y_real, axis=-1), dtype="float32")) / batch_size
            loss_c = self.cross_entropy(y_real, y_pred)
        # Calculate gradients
        grads = tape.gradient(loss_c, self.θ)
        self.optimizer_c.apply_gradients(zip(grads, self.θ))

        # Update loss trackers
        self.a_loss_tracker.update_state(loss_adv)
        self.c_loss_tracker.update_state(loss_c)
        self.acc_tracker.update_state(acc)

        return {"loss_a": self.a_loss_tracker.result(),
                "loss_c": self.c_loss_tracker.result(),
                "acc": self.acc_tracker.result()}

    def call(self, inputs, **kwargs):
        #y_logits = self.resNet(inputs)
        #return tf.keras.activations.softmax(y_logits, axis=-1)
        return inputs

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    model = AdversarialClassifier(num_classes=4,
                                  input_shape=(112, 112, 3),
                                  weights=None
                                  )

    model.summary()

    # optimizerss
    opt_a = tf.keras.optimizers.Adam(learning_rate=0.2)
    opt_c = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(adversarial_optimizer=opt_a, classifier_optimizer=opt_c, ε=3)

    # dataset = create_data_set(,
    #                           img_dim=112,
    #                           batch_size=1,
    #                           label_dict={"cow": 0, "dog": 1, "mac": 2, "tig": 3})

    data_gen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess)

    train_gen = data_gen.flow_from_directory(directory="/home/simon/PycharmProjects/AdversarialRobustness/images/",
                                             target_size=(112, 112),
                                             class_mode='categorical',
                                             batch_size=4,
                                             seed=123,
                                             subset="training"
                                             )

    history = model.fit(train_gen, epochs=100, steps_per_epoch=train_gen.n // train_gen.batch_size)



