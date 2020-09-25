import os
import tensorflow as tf
import torch

class Clerk():

    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.tensorboard_file_writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir, "train"))
        self.tensorboard_file_writer_val= tf.summary.create_file_writer(os.path.join(self.log_dir, "val"))

    def log_tensorboard_train_scalar(self, tag, scalar, epoch=0):
        with self.tensorboard_file_writer_train.as_default():
            tf.summary.scalar(tag, torch.tensor(scalar), step=epoch)
            self.tensorboard_file_writer_train.flush()

    def log_tensorboard_val_scalar(self, tag, scalar, epoch=0):
        with self.tensorboard_file_writer_val.as_default():
            tf.summary.scalar(tag, scalar, step=epoch)
            self.tensorboard_file_writer_val.flush()



