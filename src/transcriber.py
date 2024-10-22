import os
import time
import requests
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from openai import AzureOpenAI
from helper import BinaryFileReaderCallback, GenericTools
from string import Template

class AudioTranscriber:
    def __init__(self):
        load_dotenv()
        self.subscription_key = os.getenv('AZURE_SPEECH_SERVICES_KEY')
        self.region = os.getenv('AZURE_SPEECH_SERVICES_REGION')
        self.transcriptions = []

    def conversation_transcriber_transcribed_cb(self, evt: speechsdk.SpeechRecognitionEventArgs):
        print('TRANSCRIBED:')
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print('\tText={}'.format(evt.result.text))
            print('\tSpeaker ID={}'.format(evt.result.speaker_id))
            self.transcriptions.append({
                "text": evt.result.text,
                "speaker": evt.result.speaker_id
            })  # Collect the transcription results with speaker diarization
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

    def transcribe_audio_stt(self, audio_file_path, language='en-US'):        
        compressed_format = speechsdk.audio.AudioStreamFormat(compressed_stream_format=speechsdk.AudioStreamContainerFormat.ANY)
        callback = BinaryFileReaderCallback(audio_file_path)
        stream = speechsdk.audio.PullAudioInputStream(stream_format=compressed_format, pull_stream_callback=callback)

        speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_config.speech_recognition_language = language
        audio_input = speechsdk.audio.AudioConfig(stream=stream)

        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_input)
        
        self.transcriptions = []
        transcribing_stop = False

        def stop_cb(evt: speechsdk.SessionEventArgs):
            print('CLOSING on {}'.format(evt))
            nonlocal transcribing_stop
            transcribing_stop = True

        conversation_transcriber.transcribed.connect(self.conversation_transcriber_transcribed_cb)    
        conversation_transcriber.session_stopped.connect(stop_cb)
        conversation_transcriber.canceled.connect(stop_cb)

        conversation_transcriber.start_transcribing_async()

        while not transcribing_stop:
            time.sleep(.5)

        conversation_transcriber.stop_transcribing_async()

        return self.transcriptions

    def transcribe_audio_whisper(self, audio_file_path):       
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),  
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        deployment_id = "whisper" 
        
        result = client.audio.transcriptions.create(
            file=open(audio_file_path, "rb"),            
            model=deployment_id
        )
        
        return result.text

    def transcribe_audio_fast(self, audio_file_path, language='en-US'):
        config = Template('{"locales":["$language"], "profanityFilterMode": "None"}')
        
        url = os.getenv('AZURE_FAST_TRANSCRIPTION_ENDPOINT')
        payload = {'definition': config.substitute(language=language)}
        files = [('audio', (os.path.basename(audio_file_path), open(audio_file_path, 'rb'), 'audio/mpeg'))]
        headers = {
            'Ocp-Apim-Subscription-Key': os.getenv('AZURE_SPEECH_SERVICES_KEY'),
            'Accept': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        return response.json()['combinedPhrases'][0]['text']

    def transcribe_audios(self, audio_folder='audios', output_folder='transcriptions'):
        
        folder_tools = GenericTools()
        
        folder_tools.create_folder(audio_folder)
        folder_tools.create_folder(f'{output_folder}/raw')
        
        folder_tools.clean_folder(output_folder)
        folder_tools.clean_folder(f'{output_folder}/raw')

        for audio_file in os.listdir(audio_folder):
            audio_file_path = f"{audio_folder}/{audio_file}"
            
            transcriptions_stt = self.transcribe_audio_stt(audio_file_path)
            transcription_whisper = self.transcribe_audio_whisper(audio_file_path)
            transcription_fast = self.transcribe_audio_fast(audio_file_path)

            with open(f'{output_folder}/raw/stt_transcription_{audio_file}.txt', 'w') as f:
                for transcription in transcriptions_stt:
                    f.write(f"Speaker {transcription['speaker']}: {transcription['text']}\n")

            with open(f'{output_folder}/raw/whisper_transcription_{audio_file}.txt', 'w') as f:
                f.write(transcription_whisper)

            with open(f'{output_folder}/raw/fast_transcription_{audio_file}.txt', 'w') as f:
                f.write(transcription_fast)

            print(f"Transcription for {audio_file} created successfully")
    


if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcriber.transcribe_audios()
