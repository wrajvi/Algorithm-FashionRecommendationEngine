import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors

features_list =np.array(pickle.load(open('embeddings.pkl','rb')))
# print(np.array(features_list).shape)
filenames = pickle.load(open('filenames.pkl' ,'rb'))



model = ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])



def save_uploaded_file(uploded_file):
    try:
        with open(os.path.join('uploads',uploded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def feature_extraction(img_path , model):
    img = image.load_img(img_path , target_size=(224,224))
    # converting image to numpy array
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    #flatten-->to convert in 1D
    result=model.predict(preprocessed_img).flatten()

    normalized_result = result/norm(result)

    return normalized_result
def recommend(features,features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distances , indices = neighbors.kneighbors([features])
    return indices


# link for emojis https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title="Fashion Recommender" , page_icon=":eyeglasses:",layout="wide")

st.markdown("<h1 style='text-align: center; color:#35b2b0;'>Fashion Recommendation Engine</h1>", unsafe_allow_html=True)

st.image("static/intro.jpg")

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        # col1,col2,col3=st.columns([0.01,5,0.01])
        # with col1:
        #     st.write('')
        # with col2:
        #     st.image(display_image)
        # with col3:
        #     st.write('')
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        # st.text(features)
        indices = recommend(features,features_list)

        col1, col2, col3, col4, col5 =st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])

        with col2:
            st.image(filenames[indices[0][1]])

        with col3:
            st.image(filenames[indices[0][2]])

        with col4:
            st.image(filenames[indices[0][3]])

        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("Error occured in file uploading")





