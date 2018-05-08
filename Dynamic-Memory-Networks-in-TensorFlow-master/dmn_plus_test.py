from __future__ import print_function
from __future__ import division

import sys
import time

import numpy as np
from copy import deepcopy

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import test_input as test_input

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
    lr = 0.001 # was 0.001 initially
    l2 = 0.001 # increasing regularization, 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3)

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 5500  ######## COnfigurable

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)

# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """We could have used RNN for parsing sentence but that tends to overfit.
    The simpler choice would be to take sum of embedding but we loose loose positional information.
    Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

class DMN_PLUS(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.ivocab, self.entity_vocab = test_input.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.ivocab, self.entity_vocab = test_input.load_babi(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_sentences, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))
        self.speaker_info_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder))

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size) # hidden size gives the memory of the cell
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                questions,
                dtype=np.float32,
                sequence_length=self.question_len_placeholder
        )
        # print(self.question_placeholder)
        # print(q_vec)
        # print(s)
        return q_vec

    def get_input_representation(self):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        
        # print(inputs)
        # print(self.encoding.shape)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)
        # print(inputs)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_gru_cell,
                backward_gru_cell,
                inputs,
                dtype=np.float32,
                sequence_length=self.input_len_placeholder
        )

        # sum forward and backward output vectors
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # print("Manu")
        # print(self.input_placeholder)
        # print(inputs)
        # print(outputs)
        # print(fact_vecs)
        # print(s)

        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, speaker_info, reuse): 
        # fact_vec = Batch x hidden layer size
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):  

            # print("here")
            # print(q_vec)
            # print(prev_memory)
            # print(fact_vec)
            # features = [fact_vec*q_vec,
            #             fact_vec*prev_memory,
            #             tf.abs(fact_vec - q_vec),
            #             tf.abs(fact_vec - prev_memory),
            #             speaker_info]
            features = [fact_vec*q_vec,
            fact_vec*prev_memory,
            tf.abs(fact_vec - q_vec),
            tf.abs(fact_vec - prev_memory)]
            # print(features)

            feature_vec = tf.concat(features, 1)
            # print(feature_vec)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                            self.config.embed_size,
                            activation_fn=tf.nn.tanh,
                            reuse=reuse, scope="fc1")

            # print(attention)

            attention = tf.contrib.layers.fully_connected(attention,
                            1,
                            activation_fn=None,
                            reuse=reuse, scope="fc2")
            # print(attention)

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, speaker_info, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, speaker_info, bool(hop_index) or bool(i)), axis=1)  # reuse is false only for 0th hop, and first fv.
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        # print("hello")
        # print(fact_vecs, attentions)
        # print("done")
        reuse = True if hop_index > 0 else False

        speaker_info_sentence = tf.expand_dims(tf.ones([self.max_sentences,1]), 1) * speaker_info
        speaker_info_sentence = tf.transpose(speaker_info_sentence, perm=[1,0,2])  # assigning the speaker to each sentence spoken
        print("speaker_info_sentence", speaker_info_sentence)

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, speaker_info_sentence, attentions], 2)
        # gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(2*self.config.hidden_size),
                    gru_inputs,
                    dtype=np.float32,
                    sequence_length=self.input_len_placeholder
            )
        # fact_trans = tf.transpose(fact_vecs, perm=[0,2,1])
        # final = tf.matmul(fact_trans, attentions)
        # episode = tf.squeeze(final)

        print("attention : ", attentions)
        # episode = fact_vecs * attentions
        print(episode)
        return episode

    def add_answer_module(self, rnn_output, q_vec, speaker_info):
        """Linear softmax answer module"""

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
        # with tf.variable_scope("speaker_info", initializer=tf.contrib.layers.xavier_initializer()):
        # speaker_info = tf.nn.embedding_lookup(self.embeddings, self.speaker_info_placeholder)
        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                    self.vocab_size,
                    activation=None)
        # output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
        #     self.vocab_size,
        #     activation=None)

        return output

    def inference(self):
        """Performs inference on the DMN model"""

        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation()


        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation()

        with tf.variable_scope("speaker", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get speaker representation')
            speaker_info = tf.nn.embedding_lookup(self.embeddings, self.speaker_info_placeholder)        

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, speaker_info, i)

                print("episode : prev_mem : q_vec : fact_vecs", episode, prev_memory, q_vec, fact_vecs)
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                            self.config.hidden_size,
                            activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec, speaker_info)
            # output = self.add_answer_module(output, q_vec)

        return output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        print(len(data[0]))
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config.batch_size
        print (str(total_steps)+" "+str(config.batch_size))
        total_loss = []
        accuracy = 0
        final_ans = range(len(data[0]))
        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, si = data
        qp, ip, ql, il, im, a, si = qp[p], ip[p], ql[p], il[p], im[p], a[p], si[p]
        # print("Manu",len(data[0]), len(a))
        # print("total size", total_steps*config.batch_size)
        for step in range(total_steps):
            # print("here")
            # print(total_steps)
            # print(step)
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.speaker_info_placeholder: si[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = a[step*config.batch_size:(step+1)*config.batch_size]
            accuracy += np.sum(pred == answers)/float(len(answers))
            for k,pred_val in enumerate(pred):
                final_ans[step*config.batch_size + k] = self.entity_vocab[pred_val]
                print("pred_val is " + str(pred_val) + " and mapped entity is " + str(final_ans[step*config.batch_size + k]))
                print("\t the predicted ans is " + self.ivocab[pred_val] + "\n\t the correct ans is " + self.ivocab[answers[k]])
            # if(train == False) :
                # for i,ans in enumerate(answers) :
                #     if(self.ivocab[int(qp[i][0])] == "joseph"):
                #     print("######", self.ivocab[int(qp[i][0])], self.ivocab[ans])
                # print("ivocab size = ", len(self.ivocab))
                # print(tf.shape(pred), tf.shape(answers))
                # print("Sentence:")
                # for i,ans in enumerate(answers) :
                #     i += step*config.batch_size
                #     # print("#####") 
                #     # print(pred[i],ans)
                # #     for snt in ip[i]:
                # #         if(snt[0]==0):
                # #             break
                # #         for wrd in snt:
                # #             if(wrd == 0):
                # #                 break
                # #             print(self.ivocab[wrd], end = ' ')
                # #         print("")

                #     print("Question : " + self.ivocab[int(qp[i][0])])
                #     print("Answer given: "+ self.ivocab[pred[i-step*config.batch_size]] + "\t Correct Answer: "+ self.ivocab[ans])
                #     # print("Entry number: "+ str(p[i]))
                #     # print("Input", ip[i])
                #     #print(self.ivocab[int(qp[i][0])],self.ivocab[pred[i]], self.ivocab[ans], p[step*config.batch_size+i])
                #     print("__________")
            # summ = tf.Summary()
            # summ.value.add(tag='accuracy',simple_value=accuracy)
            # #tf.summary.scalar('accuracy', accuracy)
            # if train_writer is not None:
            #     train_writer.add_summary(summ, num_epoch*total_steps + step)

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                #sys.stdout.write('\r{} / {} : loss = {}'.format(
                print('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                #sys.stdout.flush()

        # variables_names = [v.name for v in tf.trainable_variables()]
        # for k in variables_names:
        #     print ("Variable: ", k)

        if verbose:
            #sys.stdout.write('\r')
            print('\r')
        
        if train == False:
            f = open('answer/log.txt', 'w')
            for val in final_ans : 
                f.write("%d\n" % val)
                print(val)
            f.close()
        # print("*****"+str(total_steps))
        return np.mean(total_loss), accuracy/float(total_steps)


    def __init__(self, config):
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()

        # set up embedding
        self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

