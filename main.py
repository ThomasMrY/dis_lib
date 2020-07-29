from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.system("pip install sklearn --user")
os.system("pip install gin-config --user")
os.system("pip install simplejson --user")
os.system("pip install tensorflow_hub>=0.2 --user")
os.system("pip install tensorflow_probability==0.7 --user")
os.system("pip install pandas --user")

import sys
sys.path.append('./')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from evaluation import evaluate
from evaluation.metrics import utils
from methods.unsupervised import train
from methods.unsupervised import vae
from postprocessing import postprocess
from utils import aggregate_results
import tensorflow as tf
import gin.tf

tf.logging.set_verbosity(tf.logging.INFO)
base_path = "example_output"
overwrite = False
path_vae = os.path.join(base_path, "vae")
train.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["model.gin"])


@gin.configurable("BottleneckVAE")  # This will allow us to reference the model.
class BottleneckVAE(vae.BaseVAE):
    """BottleneckVAE.

    The loss of this VAE-style model is given by:
    loss = reconstruction loss + gamma * |KL(app. posterior | prior) - target|
    """

    def __init__(self, gamma=gin.REQUIRED, target=gin.REQUIRED):
        self.gamma = gamma
        self.target = target

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        # This is how we customize BaseVAE. To learn more, have a look at the
        # different models in vae.py.
        del z_mean, z_logvar, z_sampled
        return self.gamma * tf.math.abs(kl_loss - self.target)

gin_bindings = [
    "model.model = @BottleneckVAE()",
    "BottleneckVAE.gamma = 4",
    "BottleneckVAE.target = 10."
]
# Call training module to train the custom model.
path_custom_vae = os.path.join(base_path, "BottleneckVAE")
train.train_with_gin(
    os.path.join(path_custom_vae, "model"), overwrite, ["model.gin"],
    gin_bindings)

for path in [path_vae, path_custom_vae]:
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                   postprocess_gin)