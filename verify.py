import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
import streamlit as st
from transcribe import *
import time
from speechbrain.pretrained import SpeakerRecognition


st.header("Trascribe Audio")

fileObject = st.file_uploader(label = "Please upload your sample audio file of the interviewee" )
fileObject = st.file_uploader(label = "upload the recording of the interviewee" )
if fileObject:
    token, t_id = upload_file(fileObject)
    result = {}
    #polling
    sleep_duration = 1
    percent_complete = 0
    progress_bar = st.progress(percent_complete)
    st.text("Currently in queue")
    while result.get("status") != "processing":
        percent_complete += sleep_duration
        time.sleep(sleep_duration)
        progress_bar.progress(percent_complete/10)
        result = get_text(token,t_id)

    sleep_duration = 0.01

    for percent in range(percent_complete,101):
        time.sleep(sleep_duration)
        progress_bar.progress(percent)

    with st.spinner("Processing....."):
        while result.get("status") != 'completed':
            result = get_text(token,t_id)

    st.balloons()
    st.header("Transcribed Text")
    st.subheader(result['text'])



verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("1_sam.mp3", "3_sam.mp3")

print(prediction, score)
