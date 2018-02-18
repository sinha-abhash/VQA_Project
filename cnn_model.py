import h5py
import tensorflow as tf
import os

feature_path = 'data/features_vgg19_2.h5'
f = h5py.File(feature_path, 'r')
filenames = [os.path.basename(file_path) for file_path in f['filenames']]
for key, value in dict(f['vgg_19']).iteritems():
    features = value

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


def read_features(img=None):
    if img:
        idx = filenames.index(img)
        feature = features[idx]
    return feature



def img_model(img_feature):

    img_fc = tf.layers.dense(img_feature, 128, activation = tf.nn.tanh)
    return img_fc

if __name__ == '__main__':
    read_features()