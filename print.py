from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# os.system("pip install sklearn --user")
# os.system("pip install gin-config --user")
# os.system("pip install simplejson --user")
# os.system("pip install tensorflow_hub>=0.2 --user")
# os.system("pip install tensorflow_probability==0.7 --user")
# os.system("pip install pandas --user")
import sys
sys.path.append('./')
from absl import app
from absl import flags
from absl import logging
from config import reproduce
from evaluation import evaluate
from methods.unsupervised import train
from postprocessing import postprocess
from visualize import visualize_model
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string("study", "unsupervised_study_v1",
                    "Name of the study.")
flags.DEFINE_string("output_directory", None,
                    "Output directory of experiments ('{model_num}' will be"
                    " replaced with the model index  and '{study}' will be"
                    " replaced with the study name if present).")
# Model flags. If the model_dir flag is set, then that directory is used and
# training is skipped.
flags.DEFINE_string("model_dir", None, "Directory to take trained model from.")
# Otherwise, the model is trained using the 'model_num'-th config in the study.
flags.DEFINE_integer("model_num", 0,
                     "Integer with model number to train.")
flags.DEFINE_boolean("only_print", False,
                     "Whether to only print the hyperparameter settings.")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")


def main(unused_argv):
  # Obtain the study to reproduce.
  study = reproduce.STUDIES[FLAGS.study]

  # Print the hyperparameter settings.
  if FLAGS.model_dir is None:
    study.print_model_config(FLAGS.model_num)
  else:
    print("Model directory (skipped training):")
    print("--")
    print(FLAGS.model_dir)
  print()
  study.print_postprocess_config()
  print()
  study.print_eval_config()
  if FLAGS.only_print:
    return


if __name__ == "__main__":
  app.run(main)