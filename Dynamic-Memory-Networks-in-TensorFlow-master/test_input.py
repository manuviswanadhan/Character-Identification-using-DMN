from __future__ import division
from __future__ import print_function
import json
import copy

import sys

import os as os
import numpy as np

import test_entity as test_entity

# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def init_babi(fname):

    data = json.load(open(fname))
    #pprint(data)
    tasks = []
    task = None
    sentences = []
    for episode in data["episodes"] :
        for scene in episode["scenes"] :
            task = {"C": "", "Q": "", "A": "", "S": ""}
            for utterance in scene["utterances"]:
                sentence = ''
                # sentence += ' '.join(s for s in utterance["speakers"])
                # sentence += ' ' 
                for i,token in enumerate(utterance["tokens"]):
                    # token += utterance["speakers"]
                    # if(len(sentence) != 0) :
                    #     sentence = sentence[:-2]
                    sentence += ' '.join(word for word in token)
                    sentence += ' $& '
                    # sentence += utterance["speakers"]
                    character_entity = utterance["character_entities"][i]
                    task["Q"] = []
                    task["A"] = []
                    for c in character_entity : 
                        if(len(c) == 3):
                            task["C"] = sentence
                            #print("hello" , token[c[0]])
                            #task["Q"] = ' '.join(word for word in token[c[0]:c[1]])
                            task["Q"] = token[c[0]]
                            task["A"] = c[2]
                            task["S"] = ' '.join(s for s in utterance["speakers"])
                            # if(token[c[0]] == "Joseph") :
                            # print("________________")
                            # print(str(len(tasks)))
                            # print("task[C]")
                            # print(task["C"])
                            # print("question: "+ task["Q"] + " \nanswer: " + task["A"])
                            # print("C = ")
                            # print(c)
                            # print("token = ")
                            # print(token)
                            # print("^^^^^^^^^^^^^^")
                            tasks.append(copy.deepcopy(task))
    return tasks


def get_babi_raw():
    babi_train_raw = init_babi('./train.json')
    babi_test_raw = init_babi('./test.json')
    babi_valid_raw = init_babi('./data/data_valid.json')
    print("hello"+ str(len(babi_test_raw))+" "+str(len(babi_train_raw)))
    return babi_train_raw, babi_valid_raw, babi_test_raw

            
def load_glove(dim):
    word2vec = {}
    
    print("==> loading glove")
    with open(("./data/glove/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print("==> glove is loaded")
    
    return word2vec

def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def process_word(word, word2vec, vocab, ivocab, entity_vocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)

        vocab[word] = next_index
        entity_vocab[next_index] = test_entity.get_index(word)
        # print("vocab of " + str(word) + " is " + str(vocab[word]) + " and its entity_vocab is " + str(entity_vocab[next_index]))
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")

def process_input(data_raw, floatX, word2vec, vocab, ivocab, entity_vocab, embed_size, split_sentences=False):
    questions = []
    inputs = []
    answers = []
    speaker_info = []
    input_masks = []

    for x in data_raw:
        # print("single x")
        # print(x) 
        if split_sentences:
            inp = x["C"].lower().split(' $& ') 
            inp = [w for w in inp if len(w) > 0]
            inp = [i.split() for i in inp]
        else:
            inp = x["C"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]

        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

    #for x in data_raw:
        # print("single x")
        # print(x) 
        # inp = x["C"]
        # q = x["Q"]
        # print(inp)
        if split_sentences: 
            inp_vector = [[process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        entity_vocab = entity_vocab,
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in s] for s in inp]
        else:
            inp_vector = [process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab,
                                        entity_vocab = entity_vocab, 
                                        word_vector_size = embed_size, 
                                        to_return = "index") for w in inp]
        # print("INPUT")
        # print(inp)
        # print(inp_vector)
                                    
        q_vector = [process_word(word = w, 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    entity_vocab = entity_vocab,
                                    word_vector_size = embed_size, 
                                    to_return = "index") for w in q]
        
        if split_sentences:
            inputs.append(inp_vector)
        else:
            inputs.append(np.vstack(inp_vector).astype(floatX))
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(process_word(word = x["A"], 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        entity_vocab = entity_vocab,
                                        word_vector_size = embed_size, 
                                        to_return = "index"))

        speaker_info.append(process_word(word = x["S"], 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        entity_vocab = entity_vocab,
                                        word_vector_size = embed_size, 
                                        to_return = "index"))
        # NOTE: here we assume the answer is one word! 

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode")

    return inputs, questions, answers, speaker_info, input_masks

def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)
    

def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(len(padded_sentences)-max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]),(0,0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)

def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding

def load_babi(config, split_sentences=False):
    vocab = {}
    ivocab = {}
    entity_vocab = {}

    babi_train_raw, babi_valid_raw, babi_test_raw = get_babi_raw()

    if config.word2vec_init:
        assert config.embed_size == 100
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}

    # set word at index zero to be end of sentence token so padding with zeros is consistent
    process_word(word = "<eos>", 
                word2vec = word2vec, 
                vocab = vocab, 
                ivocab = ivocab, 
                entity_vocab = entity_vocab,
                word_vector_size = config.embed_size, 
                to_return = "index")

    print('==> get train inputs')
    train_data = process_input(babi_train_raw, config.floatX, word2vec, vocab, ivocab, entity_vocab, config.embed_size, split_sentences)
    print('==> get validation inputs')
    valid_data = process_input(babi_valid_raw, config.floatX, word2vec, vocab, ivocab, entity_vocab, config.embed_size, split_sentences)
    print('==> get test inputs')
    test_data = process_input(babi_test_raw, config.floatX, word2vec, vocab, ivocab, entity_vocab, config.embed_size, split_sentences)
    # print("hhhhh"+str(len(test_data[1])))

    if config.word2vec_init:
        assert config.embed_size == 100
        word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init, (len(ivocab), config.embed_size))


    # full_data = [train_data[i]+valid_data[i] for i in range(len(train_data))]
    # inputs, questions, answers, speaker_info, input_masks = full_data if config.train_mode else test_data
    inputs, questions, answers, speaker_info, input_masks = train_data if config.train_mode else test_data

    print ("MODE "+str(config.train_mode))

    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)

    max_q_len = np.max(q_lens)
    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    #pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)

    answers = np.stack(answers)
    speaker_info = np.stack(speaker_info)

    test_len = int(0.8 * len(train_data[0]))
    print("test_len", test_len)

    if config.train_mode:
        print("manu",len(questions))
        train = questions[:test_len], inputs[:test_len], q_lens[:test_len], input_lens[:test_len], input_masks[:test_len], answers[:test_len], speaker_info[:test_len]

        valid = questions[test_len:], inputs[test_len:], q_lens[test_len:], input_lens[test_len:], input_masks[test_len:], answers[test_len:], speaker_info[test_len:]
        print("FINAL "+str(len(valid[0])))
        # print(ivocab)
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, len(vocab), ivocab #, entity_vocab

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers, speaker_info
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, len(vocab), ivocab #, entity_vocab
