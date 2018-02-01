import tflearn
import tensorflow as tf


def word2vecmodel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    print "Creating word2vec model..."
    
    embedding_m = tf.constant_initializer(embedding_matrix)
    #net = tflearn.input_data(shape=[seq_length, 300])
    word_input = tf.placeholder(dtype=tf.float32, shape=[None, 300])
    net = tflearn.input_data(placeholder=word_input)

    # create embedding weights, set trainable to False, so weights are not updated
    #net = tflearn.embedding(net, input_dim=num_words, output_dim=embedding_dim, weights_init=[embedding_matrix], trainable=False, name="EmbeddingLayer")
    net = tflearn.embedding(net, input_dim=num_words, output_dim=embedding_dim, weights_init=embedding_m, trainable=False, name="EmbeddingLayer")
    net = tflearn.lstm(net, 128, return_seq=False)
    net = tflearn.dropout(net, dropout_rate)
    '''net = tflearn.lstm(net, 256, return_seq=True)
    net = tflearn.dropout(net, dropout_rate)
    net = tflearn.lstm(net, 128, return_seq=False)'''

    return net
