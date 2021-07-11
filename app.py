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

import pyrebase
import os

#-------------------------------------------------------------------------------
# Variables & Setup

# filelist = [ f for f in os.listdir(".") if f.endswith(".JPG") ]
# for f in filelist:
#     os.remove(os.path.join(".", f))

# config = {
# "apiKey": "AIzaSyDFyW8s4L8pabax_r9QajAkfxaJBLB00AE",
# "authDomain": "fileupload-962b1.firebaseapp.com",
# "databaseURL": "https://fileupload-962b1.firebaseio.com",
# "projectId": "fileupload-962b1",
# "storageBucket": "fileupload-962b1.appspot.com",
# "serviceAccount": "fileupload-962b1-firebase-adminsdk-tnjsb-72bf80e9c9.json"
# }

# firebase_storage = pyrebase.initialize_app(config)
# storage = firebase_storage.storage()

#-------------------------------------------------------------------------------
# Uploading And Downloading Images

# storage.child("Guitar.JPG").put("Guitar.JPG")
# storage.child("PlayingGuitar.JPG").download("PlayingGuitar.JPG")


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

# storage.child(fileObject.name).put(fileObject.name)
# storage.child(fileObject2.name).put(fileObject2.name)



# bucket_name = "fileupload-962b1.appspot.com"


# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="fileupload-962b1-firebase-adminsdk-tnjsb-72bf80e9c9.json"


# def list_blobs(bucket_name):
#     """Lists all the blobs in the bucket."""
    
    
#     storage_client = storage.Client()

#     # Note: Client.list_blobs requires at least package version 1.17.0.
#     blobs = storage_client.list_blobs(bucket_name)

#     for blob in blobs:
#         print(blob.name)


# list_blobs(bucket_name)

# def get_blob_path(blob):
#         """
#         Gets blob path.
#         :param blob: instance of :class:`google.cloud.storage.Blob`.
#         :return: path string.
#         """
#         return bucket_name + "/" + blob.name 




if fileObject and fileObject2 is not None:
    file_details = {"FileName":fileObject.name,"FileType":fileObject.type,"FileSize":fileObject.size}
    st.write(file_details)
    st.write(fileObject)
    st.audio(fileObject, format='audio/ogg')
    st.audio(fileObject2, format='audio/ogg')

    
        
    if st.button('result'):
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        score, prediction = verification.verify_files(get_blob_path(fileObject),get_blob_path(fileObject2))
        
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        transcription = asr_model.transcribe_file(get_blob_path(fileObject2))
        st.write(prediction)
        st.write(score)
    






    
