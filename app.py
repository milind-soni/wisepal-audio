import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
import streamlit as st
import time
from speechbrain.pretrained import SpeakerRecognition
import os
import wave
from google.cloud import storage
from speechbrain.pretrained import EncoderDecoderASR
from google.cloud import storage
from google.cloud import pubsub_v1
from scipy.io import wavfile
from pydub import AudioSegment
import pyrebase
import os

from google.cloud import storage
bucket_name = "fileupload-962b1.appspot.com"


config = {
"apiKey": "AIzaSyDFyW8s4L8pabax_r9QajAkfxaJBLB00AE",
"authDomain": "fileupload-962b1.firebaseapp.com",
"databaseURL": "https://fileupload-962b1.firebaseio.com",
"projectId": "fileupload-962b1",
"storageBucket": "fileupload-962b1.appspot.com",
"serviceAccount": "fileupload-962b1-firebase-adminsdk-tnjsb-72bf80e9c9.json"
}

firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="fileupload-962b1-firebase-adminsdk-tnjsb-72bf80e9c9.json"


def get_blob_path(blob):
        """
        Gets blob path.
        :param blob: instance of :class:`google.cloud.storage.Blob`.
        :return: path string.
        """
        return bucket_name + "/" + blob.name 

path = os.path.dirname(__file__)

def save_uploadedfile(uploaded_file):
    #  file_var = AudioSegment.from_mp3(uploaded_file) 
    
     storage.child(uploaded_file).put(uploaded_file)
    #  file_var.export(path+ "/uploads/" + uploaded_file)





st.header("Audio Verification app")
st.set_option('deprecation.showfileUploaderEncoding', False)





prediction = None 
score = None 

fileObject = st.file_uploader(label = "Please upload your sample audio file of the interviewee" )

fileObject2 = st.file_uploader(label = "Please upload your sample audio file of the interviewee" ,key = "2" )



if fileObject and fileObject2 is not None:
    file1_details = {"FileName":fileObject.name,"FileType":fileObject.type}
    file2_details = {"FileName":fileObject2.name,"FileType":fileObject2.type}
    save_uploadedfile(fileObject.name)
    save_uploadedfile(fileObject2.name)
    
    if st.button('result'):
        
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        score, prediction = verification.verify_files(bucket_name +'/'+ fileObject.name,bucket_name +'/' fileObject2.name)
        st.write(prediction)
        st.write(score)
            
#     storage.child(fileObject.name).put(fileObject.name)
#     storage.child(fileObject2.name).put(fileObject2.name)

    
        
#     asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
#     transcription = asr_model.transcribe_file(get_blob_path(fileObject2))
    
    
