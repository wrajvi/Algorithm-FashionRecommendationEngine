import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# creating the resnet modle
# include_top=False-->I am adding my own top layer
model = ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
# using Already trained model
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    # added the top layer
    GlobalMaxPooling2D()
])

# print(model.summary())

def extract_features(img_path , model):
    img = image.load_img(img_path , target_size=(224,224))
    # converting image to numpy array
    img_array = image.img_to_array(img)
    # keras work on batches of images
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    #flatten-->to convert in 1D
    result=model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

# putting all images in a list
filenames = []
# print(os.listdir('images'))
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))
# print(len(filenames))
# print(filenames[0:5])
# putting the features of each images in a list
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Extracting All features of All IMAGES in embeddings
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

