import os
from string import Template
from helper import AzureOpenAI, GenericTools
import prompts as prt

class TranscriptionAdjuster:
    def __init__(self, folder):
        self.folder_source = folder
        self.folder_adjusted = self.folder_source.replace(self.folder_source.split('/')[-1:][0], 'adjusted')
        self.user_prompt = prt.user_prompt_transcription_adjuster       

    def adjust_transcriptions(self):

        generic_tools = GenericTools()
        
        generic_tools.create_folder(self.folder_adjusted)        
        generic_tools.clean_folder(self.folder_adjusted)

        for file in os.listdir(self.folder_source):
            if file.endswith(".txt"):
                transcription = self._read_file(file)
                prompt = self._create_prompt(transcription)
                result = self._send_request(prompt)
                self._write_adjusted_transcription(file, result)

    def _read_file(self, file):
        with open(f"{self.folder_source}/{file}", "r") as f:
            return f.read()

    def _create_prompt(self, transcription):
        template = Template(self.user_prompt)
        return template.substitute(transcription=transcription)

    def _send_request(self, prompt):
        return AzureOpenAI().send_llm_request(prt.system_prompt_transcription_adjuster, prompt, return_json=False)

    def _write_adjusted_transcription(self, file, result):
        with open(f"{self.folder_adjusted}/{file}", "w") as f:
            f.write(result)

if __name__ == '__main__':
    adjuster = TranscriptionAdjuster('transcriptions/raw')
    adjuster.adjust_transcriptions()