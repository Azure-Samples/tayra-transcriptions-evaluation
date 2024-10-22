# streamlit_app.py

import streamlit as st
import os
from transcriber import AudioTranscriber
from adjuster import TranscriptionAdjuster
from evaluator import TranscriptionEvaluator

# Initialize the Streamlit app
st.title("Tayra Project")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Transcription", "Adjustment", "Evaluation"])

# Helper functions
def display_files(folder, file_type=".txt"):
    files = [f for f in os.listdir(folder) if f.endswith(file_type)]
    for file in files:
        with open(os.path.join(folder, file), "r") as f:
            st.text(f.read())

def transcribe_audio(audio_file):
    transcriber = AudioTranscriber()
    transcriptions_stt = transcriber.transcribe_audio_stt(audio_file)
    transcription_whisper = transcriber.transcribe_audio_whisper(audio_file)
    transcription_fast = transcriber.transcribe_audio_fast(audio_file)
    return transcriptions_stt, transcription_whisper, transcription_fast

def adjust_transcriptions(folder):
    adjuster = TranscriptionAdjuster(folder)
    adjuster.adjust_transcriptions()

def evaluate_transcriptions(folder_groundtruth, folder_transcriptions):
    evaluator = TranscriptionEvaluator(folder_groundtruth)
    scores = evaluator.calculate_llm_score(folder_transcriptions)
    return scores

# Transcription Section
if options == "Transcription":
    st.header("Transcription")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if audio_file:
        st.audio(audio_file)
        transcriptions_stt, transcription_whisper, transcription_fast = transcribe_audio(audio_file)
        st.subheader("STT Transcription")
        st.text(transcriptions_stt)
        st.subheader("Whisper Transcription")
        st.text(transcription_whisper)
        st.subheader("Fast Transcription")
        st.text(transcription_fast)

# Adjustment Section
elif options == "Adjustment":
    st.header("Adjustment")
    folder = st.text_input("Enter the folder path for raw transcriptions")
    if st.button("Adjust Transcriptions"):
        adjust_transcriptions(folder)
        st.success("Transcriptions adjusted successfully!")
        st.subheader("Adjusted Transcriptions")
        display_files(folder.replace("raw", "adjusted"))

# Evaluation Section
elif options == "Evaluation":
    st.header("Evaluation")
    folder_groundtruth = st.text_input("Enter the folder path for groundtruth transcriptions")
    folder_transcriptions = st.text_input("Enter the folder path for transcriptions to evaluate")
    if st.button("Evaluate Transcriptions"):
        scores = evaluate_transcriptions(folder_groundtruth, folder_transcriptions)
        st.success("Transcriptions evaluated successfully!")
        st.subheader("Evaluation Scores")
        st.write(scores)