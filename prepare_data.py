import numpy as np
import json
import h5py
import os
from constants import *
from tflearn.data_utils import to_categorical

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
    return v

def read_data(data_limit):
    print "Reading Data..."
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)
  
    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:data_limit]
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

    #shifting padding to left side
    ques_train = np.array(ques_data['ques_train'])[:data_limit, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:data_limit]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    # NOTE should've consturcted one-hots using exhausitve list of answers, cause some answers may not be in dataset
    # To temporarily rectify this, all those answer indices is set to 1 in validation set
    #train_y = to_categorical(ques_data['answers'])[:data_limit, :]
    train_y = ques_data['answers']

    return train_X, train_y

def get_metadata():
    meta_data = json.load(open(data_prepo_meta, 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

def prepare_embeddings(num_words, embedding_dim, metadata):
    if os.path.exists(embedding_matrix_filename):
        with h5py.File(embedding_matrix_filename) as f:
            return np.array(f['embedding_matrix'])

    print "Embedding Data..."
    
    embeddings_index = {}
    with open(glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, embedding_dim))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
   
    with h5py.File(embedding_matrix_filename, 'w') as f:
        f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix

def get_obj_features(json_path):
	with open(json_path + "/5.json") as data_file:
		data = json.load(data_file)

	return data
		
