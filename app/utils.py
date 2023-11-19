import tensorflow as tf
from typing import List
import cv2
import os 
import tempfile
import shutil
import subprocess
from moviepy.editor import VideoFileClip

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)



def convert_mpeg_to_mp4(mpeg_path):
    video_clip = VideoFileClip(mpeg_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        video_clip.write_videofile(temp_file.name, codec="libx264", audio_codec="aac", threads=4, logger=None)

    return temp_file.name

def load_video(path:str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    
    file_name = os.path.splitext(os.path.basename(path))[0]

    video_path = os.path.join('app', 'data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('app', 'data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments
