from __future__ import print_function
from __future__ import division

import tensorflow as tf

import time
import argparse
import os
import importlib
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-l2", "--l2_loss", type=float, default=0.001, help="specify l2 loss constant")
parser.add_argument("-lr", "--lr", type=float, default=0.001, help="specify learning rate constant")
parser.add_argument("-n", "--num_runs", type=int, help="specify the number of model runs")
parser.add_argument("-m", "--model_arch", type=str, default="none", help="specify the architecture (gru/ans/ansgru) for the model")
parser.add_argument("-sgru", "--speaker_gru", type=str, default="false", help="specify the speaker info should be fed into the attention GRU")

#parser.add_argument("-sgru", "--speaker_gru", type=str, default = "false", help="specify the speaker info should be fed into the attention GRU")
parser.add_argument("-sop", "--speaker_output", type=str, help="specify the speaker info should be fed into the output layer")
parser.add_argument("-attn", "--attn", type=str, default="false", help="specify whether attention includes speaker info (true)")
parser.add_argument("-GRU", "--GRU", type=str, default="true", help="specify whether to use not to use specific attention GRU (false)")


args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

file_name = "dmn_master" #+ (str(args.model_arch) if args.model_arch is not None else "plus") + ( "attn" if args.attn is not "" else "")

mod = importlib.import_module(file_name)
config = mod.Config


# if dmn_type == "plus":
#     from mod import Config

#     config = Config()
# else:
#     raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.babi_id = args.babi_task_id if args.babi_task_id is not None else str(1)
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1
config.attn = args.attn
config.lr = args.lr if args.lr is not None else 0.001
config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
config.GRU = args.GRU
config.attn = args.attn
config.speaker_gru = args.speaker_gru
config.speaker_output = args.speaker_output

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "plus":
        #from dmn_plus import DMN_PLUS
        DMN_PLUS = mod.DMN_PLUS

        model = DMN_PLUS(config)

#postfix = (args.model_arch if args.model_arch is not None else "plus") + ( "attn" if args.attn is not "false" else "")
postfix = ("gru" if args.speaker_gru else "") + ("ans" if args.speaker_output else "") + "_" + (str(args.lr)) + "_" + str(args.l2_loss) + str(args.GRU)

#postfix = postfix + "_" + str(model.config.lr) + "_" + str(model.config.l2) + "_" + str(model.config.GRU)
print("##################postfix: " + postfix)

for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:

        sum_dir = 'summaries/train/' + postfix
        shutil.rmtree(sum_dir, ignore_errors=True, onerror=None)
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if args.restore:
            print('==> restoring weights')
            saver.restore(session, 'weights/task' + postfix + '.weights')

        print('==> starting training')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy = model.run_epoch(
                session, model.train, epoch, train_writer,
                train_op=model.train_step, train=True)
            print("Validation Acc")
            valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
            print('Training loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(valid_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            print('Vaildation accuracy: {}'.format(valid_accuracy))

            summ = tf.Summary()
            summ.value.add(tag='training loss',simple_value=train_loss)
            summ.value.add(tag='training accuracy',simple_value=train_accuracy)
            summ.value.add(tag='validation loss',simple_value=valid_loss)
            summ.value.add(tag='validation accuracy',simple_value=valid_accuracy)

            #tf.summary.scalar('accuracy', accuracy)
            if train_writer is not None:
                train_writer.add_summary(summ, epoch)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, 'weights/task' + postfix + '.weights')

            # anneal
            if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                model.config.lr /= model.config.anneal_by
                print('annealed lr to %f' % model.config.lr)

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)
