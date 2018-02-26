import tflearn
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper, MultiRNNCell


def word2vecmodel(embedding_matrix, num_words, embedding_dim):
    print "Creating word2vec model..."
    dropout_rate = 0.5
    max_words_q = 26
    question = tf.placeholder(tf.int32, [None, max_words_q], name="question")
    embeddings_var = tf.Variable(tf.random_uniform([num_words, embedding_dim], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, question)

    embedding_m = tf.constant_initializer(embedding_matrix)
    #net = tflearn.input_data(shape=[seq_length, 300])

    #net = tflearn.input_data(placeholder=word_input)

    # create embedding weights, set trainable to False, so weights are not updated
    #net = tflearn.embedding(net, input_dim=num_words, output_dim=embedding_dim, weights_init=[embedding_matrix], trainable=False, name="EmbeddingLayer")
    #net = tflearn.embedding(net, input_dim=num_words, output_dim=embedding_dim, weights_init=embedding_m, trainable=False, name="EmbeddingLayer")
    net = tflearn.lstm(batch_embedded, 128, return_seq=False)
    net = tflearn.dropout(net, dropout_rate)
    '''net = tflearn.lstm(net, 256, return_seq=True)
    net = tflearn.dropout(net, dropout_rate)
    net = tflearn.lstm(net, 128, return_seq=False)'''

    return net, question

def word2vec(vocabulary_size):
    max_words_q = 26
    input_embedding_size = 300
    rnn_size = 512
    drop_out_rate = 0.5
    rnn_layer = 2
    dim_hidden = 1024
    embed_ques_W = tf.Variable(tf.random_uniform([vocabulary_size, input_embedding_size], -0.08, 0.08),
                                    name='embed_ques_W')

    # encoder: RNN body
    lstm_1 = LSTMCell(rnn_size, input_embedding_size)
    lstm_dropout_1 = DropoutWrapper(lstm_1, output_keep_prob=1 - drop_out_rate)
    lstm_2 = LSTMCell(rnn_size, rnn_size)
    lstm_dropout_2 = DropoutWrapper(lstm_2, output_keep_prob=1 - drop_out_rate)

    embed_state_W = tf.Variable(tf.random_uniform([2 * rnn_size * rnn_layer, dim_hidden], -0.08, 0.08),
                                     name='embed_state_W')
    embed_state_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_state_b')


    stacked_lstm = MultiRNNCell([lstm_dropout_1, lstm_dropout_2])

    question = tf.placeholder(tf.int32, [20280, max_words_q])
    state = tf.zeros([20280, stacked_lstm.state_size])
    for i in range(max_words_q):
        if i == 0:
            ques_emb_linear = tf.zeros([20280, input_embedding_size])
        else:
            tf.get_variable_scope().reuse_variables()
            ques_emb_linear = tf.nn.embedding_lookup(embed_ques_W, question[:, i - 1])

        ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1 - drop_out_rate)
        ques_emb = tf.tanh(ques_emb_drop)

        output, state = stacked_lstm(ques_emb, state)

        # multimodal (fusing question & image)
        state_drop = tf.nn.dropout(state, 1 - drop_out_rate)
    state_linear = tf.nn.xw_plus_b(state_drop, embed_state_W, embed_state_b)
    state_emb = tf.tanh(state_linear)

    return state_emb, question