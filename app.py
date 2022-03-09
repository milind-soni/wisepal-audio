import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
import streamlit as st
import time
from speechbrain.pretrained import SpeakerRecognition
import os
import wave
from speechbrain.pretrained import EncoderDecoderASR
from scipy.io import wavfile
from pydub import AudioSegment
import os

# from google.cloud import storage
# bucket_name = "fileupload-962b1.appspot.com"


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

# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="fileupload-962b1-firebase-adminsdk-tnjsb-72bf80e9c9.json"
# '''
# def list_blobs(bucket_name):
#     """Lists all the blobs in the bucket."""
    

#     storage_client = storage.Client()
    
#     # Note: Client.list_blobs requires at least package version 1.17.0.
#     blobs = storage_client.list_blobs(bucket_name)

#     for blob in blobs:
#         print(bucket_name+'/' + blob.name)


# list_blobs(bucket_name)
# '''

# def get_blob_path(blob):
#         """
#         Gets blob path.
#         :param blob: instance of :class:`google.cloud.storage.Blob`.
#         :return: path string.
#         """
#         return bucket_name + "/" + blob.name 

path = os.path.dirname(__file__)

def save_uploadedfile(uploaded_file):
     file_var = AudioSegment.from_mp3(uploaded_file) 
    

     file_var.export(path+ "/uploads/" + uploaded_file)





st.header("Audio Verification app")
st.set_option('deprecation.showfileUploaderEncoding', False)


def savefiletolocal(uploaded_file,some_bytes):
    binary_file = open(path+ "/uploads/" + uploaded_file, "wb")
  
    # Write bytes to file
    binary_file.write(some_bytes)
  
    # Close file
    binary_file.close()



prediction = None 
score = None 

fileObject = st.file_uploader(label = "Please upload your sample audio file of the interviewee" )

fileObject2 = st.file_uploader(label = "Please upload your sample audio file of the interviewee" ,key = "2" )



if fileObject and fileObject2 is not None:
    file1_details = {"FileName":fileObject.name,"FileType":fileObject.type}
    file2_details = {"FileName":fileObject2.name,"FileType":fileObject2.type}
    st.audio(fileObject, format='audio/ogg')
    st.audio(fileObject2, format='audio/ogg')
    bytes_data = fileObject.getvalue()
    bytes_data2 = fileObject2.getvalue()
    #st.write(bytes_data)
    if st.button('result'):
        savefiletolocal(fileObject.name,bytes_data)
        savefiletolocal(fileObject2.name,bytes_data2)
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        score, prediction = verification.verify_files(path+ "/uploads/" + fileObject.name,path+ "/uploads/" + fileObject2.name)
               
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        transcription = asr_model.transcribe_file(path+ "/uploads/" + fileObject2.name)
    
    
        st.write(prediction)
        st.write(score)
        st.write(transcription)    
#     storage.child(fileObject.name).put(fileObject.name)
#     storage.child(fileObject2.name).put(fileObject2.name)

  
