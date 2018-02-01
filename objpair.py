from preprocessing import get_obj_pairwise, get_pairs
import tensorflow as tf
import tflearn

from prepare_data import *
from word_emb import word2vecmodel
#from .AttentionGRUCell import AttentionGRUCell
from VQA.AttentionGRUCell import AttentionGRUCell
from cnn_model import *


def get_attention(embedding_dim, query, prev_memory, fact, reuse=False):
    """Use question vector and previous memory to create scalar attention for current fact"""
    with tf.variable_scope("attention", reuse=reuse):
        features = [fact * query,
                    fact * prev_memory,
                    tf.abs(fact - query),
                    tf.abs(fact - prev_memory)]

        feature_vec = tf.concat(features, 1)

        attention = tf.contrib.layers.fully_connected(feature_vec,
                                                      embedding_dim,
                                                      activation_fn=tf.nn.tanh,
                                                      reuse=reuse, scope="fc1")

        attention = tf.contrib.layers.fully_connected(attention,
                                                      1,
                                                      activation_fn=None,
                                                      reuse=reuse, scope="fc2")

    return attention


def generate_episode(memory, query, facts, hop_index, hidden_size, input_lengths, embedding_dim):
    """Generate episode by applying attention to current fact vectors through a modified GRU"""

    attentions = [tf.squeeze(
        get_attention(embedding_dim, query, memory, fv, bool(hop_index) or bool(i)), axis=1)
        for i, fv in enumerate(tf.unstack(facts, axis=1))]

    attentions = tf.transpose(tf.stack(attentions))
    attention_softmax = tf.nn.softmax(attentions)

    attentions = tf.expand_dims(attention_softmax, axis=-1)

    reuse = True if hop_index > 0 else False

    # concatenate fact vectors and attentions for input into attGRU
    gru_inputs = tf.concat([facts, attentions], 2)

    with tf.variable_scope('attention_gru', reuse=reuse):
        _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(hidden_size),
                                       gru_inputs,
                                       dtype=np.float32,
                                       sequence_length=input_lengths)
    return episode, attention_softmax


def build_attention_model(word_emb_net, obj_fc_pair, count_features_pair_tensor, img_model):
    attentions = []

    sentiment_memories = [word_emb_net]

    # memory module
    with tf.variable_scope("memory",
                           initializer=tf.contrib.layers.xavier_initializer()):
        print('==> build episodic memory')

        # generate n_hops episodes
        prev_memory = word_emb_net

        for i in range(2):
            # get a new episode
            print('==> generating episode', i)
            episode, attn = generate_episode(prev_memory, word_emb_net, obj_fc_pair, i,
                                                 128, count_features_pair_tensor,
                                                 300)
            attentions.append(attn)
            # untied weights for memory update
            with tf.variable_scope("hop_%d" % i):
                prev_memory = tf.layers.dense(tf.concat([prev_memory, episode,
                                                         word_emb_net], 1),
                                              128,
                                              activation=tf.nn.relu)
                sentiment_memories.append(prev_memory)
        output = tf.concat([prev_memory, word_emb_net, img_model], 1)
        output_fc = tflearn.fully_connected(output, 128)

    return output_fc


def def_model(obj_shape, obj_features_tensor, word_emb_net, obj_coord_tensor, coord_shape):
    # get dense representation of object features pairwise. the output will be of size 128

    obj_features = tf.split(obj_features_tensor, obj_shape[1]/2, axis=1)
    obj_coord = tf.split(obj_coord_tensor, coord_shape[1] / 2, axis=1)
    obj_fc_pair = []
    for feature_pair, coord_pair in zip(obj_features, obj_coord):
        concat_feature = tf.reshape(feature_pair, [-1, 2*4096])
        concat_coord = tf.reshape(coord_pair, [-1, 2*4])

        input_obj1 = tflearn.input_data(shape=[None, 2*4096])
        fc_obj1 = tflearn.fully_connected(input_obj1, 128, activation="tanh")
        '''
        fc_obj_word_1 = tflearn.merge([fc_obj1, word_emb_net], mode="concat")
        fc_obj_word_2 = tflearn.fully_connected(fc_obj_word_1, 128, activation="tanh")
        '''
        input_coord1 = tflearn.input_data(shape=[None, 2*4096])
        fc_coord1 = tflearn.fully_connected(input_coord1, 128, activation="tanh")

        fc_obj_word_coord_1 = tflearn.merge([fc_obj1, fc_coord1], mode="concat")
        fc_obj_word_coord_2 = tflearn.fully_connected(fc_obj_word_coord_1, 128, activation="tanh")

        obj_fc_pair.append(fc_obj_word_coord_2)
    print ("length of obj_fc_pair list: %d" %(len(obj_fc_pair)))
    obj_fc_pair = tf.stack(obj_fc_pair)

    print(obj_fc_pair.get_shape().as_list())
    obj_fc_pair = tf.transpose(obj_fc_pair, perm=[1,0,2])
    return obj_fc_pair


def get_embed_model(dropout_rate, model_weights_filename):
    print "Creating Model..."
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())
    print "Number of classes = ", num_classes

    #embedding_matrix = get_question_features(u"Where is the dog?")
    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)

    model = word2vecmodel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if os.path.exists(model_weights_filename):
        print "Loading Weights..."
        model.load_weights(model_weights_filename)

    return model


if __name__ == '__main__':
    obj_features, obj_coord_pair, count_features_pair_list = get_obj_pairwise()


    obj_features = obj_features.reshape(obj_features.shape[0], obj_features.shape[1]*obj_features.shape[2], obj_features.shape[-1])
    obj_coord_pair = obj_coord_pair.reshape(obj_coord_pair.shape[0], obj_coord_pair.shape[1]*obj_coord_pair.shape[2], obj_coord_pair.shape[-1])

    print(obj_features.shape, obj_coord_pair.shape, count_features_pair_list.shape)
    obj_features_tensor = tf.placeholder(tf.float32, shape=obj_features.shape)
    obj_coord_tensor = tf.placeholder(tf.float32, shape=obj_coord_pair.shape)
    count_features_pair_tensor = tf.placeholder(tf.int32, shape=count_features_pair_list.shape)
    print(count_features_pair_tensor.get_shape().as_list())

    dropout_rate = 0.5
    word_emb_net = get_embed_model(dropout_rate, '')
    #keras_autoencoder(train_x)
    #test(train_x)
    #obj_pair_model(train_x)
    #tflearndensenetwork(train_x)
    #obj_coord_pair = objcoordpair.get_obj_coordinates_pair()
    obj_fc_pair = def_model(obj_features.shape, obj_features_tensor, word_emb_net, obj_coord_tensor, obj_coord_pair.shape)
    img_model = img_model()
    attention_output = build_attention_model(word_emb_net, obj_fc_pair, count_features_pair_tensor, img_model)
    '''
    y_ = tf.placeholder(dtype=tf.float32, shape=1000)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(attention_output), y_)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(125):
            obj_pair_features, obj_pair_coord, count_features_pair_list= get_pairs(i)

    '''

    print(attention_output.get_shape().as_list())

    '''
    model = tflearn.DNN(model, checkpoint_path='model', tensorboard_dir='tensorboard/tflearn_logs/',
                        best_checkpoint_path="checkpoints/", max_checkpoints=10, tensorboard_verbose=3,
                        clip_gradients=0.)

    
    model.fit([obj_features, obj_coord_pair], y,
              n_epoch=3, validation_set=0.1,
              snapshot_epoch=False, snapshot_step=400,
              show_metric=True, batch_size=64, shuffle=True,
              run_id='model_id')

    '''


    '''
    TODO
    for que_ans pairs, for every answers, prepare its hot encoding over 1000 (Answer) classes + 1 (for unknown question)
    in every image, there will be certain number of objects instance pairs. Every instance pair gets all the questions related to that image.
    this means, from 1 json file, result will have (no of images in a file, no of pairs for every image, (no of questions, no of answers) )
    '''