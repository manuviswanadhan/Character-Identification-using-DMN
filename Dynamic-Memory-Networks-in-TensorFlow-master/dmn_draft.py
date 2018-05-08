from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import importlib


import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-l2", "--l2_loss", type=float, default=0.001, help="specify l2 loss constant")
parser.add_argument("-lr", "--lr", type=float, default=0.001, help="specify learning rate constant")
parser.add_argument("-m", "--model_arch", type=str, help="specify the architecture (gru/ans/ansgru) for the model")

args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

file_name = "dmn_"+ (str(args.model_arch) if args.model_arch is not None else "plus_test")

mod = importlib.import_module(file_name)
config = mod.Config

config.lr = args.lr if args.lr is not None else 0.001
config.l2 = args.l2_loss if args.l2_loss is not None else 0.001

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.strong_supervision = False
config.model_arch = args.model_arch if args.model_arch is not None else "plus"

config.train_mode = False

print( 'Testing DMN ' + dmn_type + ' on babi task', config.babi_id)

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "plus":
        #from dmn_plus import DMN_PLUS
        DMN_PLUS = mod.DMN_PLUS

        model = DMN_PLUS(config)

postfix = config.model_arch + "_" + str(model.config.lr) + "_" + str(model.config.l2)
print("##################postfix: " + postfix)

print('==> initializing variables')
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print('==> restoring weights')
    saver.restore(session, 'weights/task' + postfix + '.weights')

    print('==> running DMN')
    test_loss, test_accuracy = model.run_epoch(session, model.test)

    print('')
    print('Test accuracy:', test_accuracy)
