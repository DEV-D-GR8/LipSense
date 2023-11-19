import streamlit as st
import os 
import tensorflow as tf 
from utils import load_data, num_to_char, convert_mpeg_to_mp4
from modelutil import load_model

st.set_page_config(layout='wide')

with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipSense')
    st.info('Experience the power of cutting-edge technology as we bring you a unique solution for bridging communication gaps. Our deep learning model can interpret lip movements in real-time, unlocking a new dimension of inclusivity for individuals with hearing impairments.')

st.title('LipSense Project') 
data_dir = os.path.join('app', 'data', 's1')
options = os.listdir(data_dir)
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options: 

    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('app', 'data', 's1', selected_video)
        
        temp_mp4_file = convert_mpeg_to_mp4(file_path)
        video = open(temp_mp4_file, 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
        
        yhat = None
        decoder = None
        
        
