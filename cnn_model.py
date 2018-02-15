import h5py
import tensorflow as tf

feature_path = 'data/features_vgg19_2.h5'

'''
def get_cnn_features_list():
    train_path = "/home/surajit/Documents/Project/VQA_Project/VQA/dataset/train_images/"
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224, 3))
    features_list = []
    for file in os.listdir(train_path):
        file = "dataset/train_images/" + file
        img = image.load_img(file, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = base_model.predict(img)
        features_list.append(feature)
    return features_list

features_list = get_cnn_features_list()
np_features = np.array(features_list)
print(np_features.shape)
'''


def read_features():
    f = h5py.File(feature_path, 'r')
    filenames = f['filenames']
    for key, value in dict(f['vgg_19']).iteritems():
        features = value


def img_model():
    img_feature = tf.placeholder(tf.float32, shape=[None, 4096])
    img_fc = tf.layers.dense(img_feature, 128, activation = tf.nn.tanh())
    return img_fc

if __name__ == '__main__':
    read_features()