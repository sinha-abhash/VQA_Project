from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D
from keras.models import Sequential, Model
from PIL import Image
import numpy as np
import os

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

def img_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(400,400, 3))
    '''    
    inputs = Input(shape=(400,400,3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(128))
    '''
    top_model = GlobalAveragePooling2D()(base_model.output)
    top_model = Dense(128, activation='relu')(top_model)

    model = Model(inputs=base_model.input, outputs=top_model)

    print(model.summary())
    return model.layers[-1].output