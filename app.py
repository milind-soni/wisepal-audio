import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
import streamlit as st
import time
from speechbrain.pretrained import SpeakerRecognition
import os
import wave

st.header("Audio Verification app")
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache
def load_audio(audio_file):
    audio_file = open(audio_file, 'rb')
    audio_bytes = audio_file.read()
    return audio_file

prediction = None 
score = None 

fileObject = st.file_uploader(label = "Please upload your sample audio file of the interviewee" )

fileObject2 = st.file_uploader(label = "Please upload your sample audio file of the interviewee" ,key = "2" )



if fileObject is not None:
    file_details = {"FileName":fileObject.name,"FileType":fileObject.type,"FileSize":fileObject.size}
    st.write(file_details)
    st.write(fileObject)
    st.audio(fileObject, format='audio/ogg')
    
if fileObject2 is not None:
    file_details = {"FileName":fileObject2.name,"FileType":fileObject2.type,"FileSize":fileObject2.size}
    st.write(file_details)
    st.write(fileObject2)
    st.audio(fileObject2, format='audio/ogg')
        
if st.button('result'):
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    score, prediction = verification.verify_files(fileObject.name, fileObject2.name)
    st.write(prediction)
    st.write(score)
  
 
    
