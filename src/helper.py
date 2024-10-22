import os, requests
import pandas as pd
import azure.cognitiveservices.speech as speechsdk

class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
    def __init__(self, filename: str):
        super().__init__()
        self._file_h = open(filename, "rb")

    def read(self, buffer: memoryview) -> int:
        try:
            size = buffer.nbytes
            frames = self._file_h.read(size)

            buffer[:len(frames)] = frames

            return len(frames)
        except Exception as ex:
            raise

    def close(self) -> None:
        print('closing file')
        try:
            self._file_h.close()
        except Exception as ex:
            print('Exception in `close`: {}'.format(ex))
            raise

class AzureOpenAI():
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.model = os.getenv("AZURE_OPENAI_MODEL")
        self.embeddings_model = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL")

    def send_llm_request(self, system_prompt, prompt, return_json=True):
        if not self.api_key or not self.endpoint or not self.model:
            raise ValueError("API key, endpoint, or model not found in environment variables")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        payload = {
                    "messages": [
                        {
                        "role": "system",
                        "content": [
                            {
                            "type": "text",
                            "text": system_prompt
                            }
                        ]
                        },
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": prompt
                            }
                        ]
                        }
                    ],
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" } if return_json else { "type": "text" }
                    }

        response = requests.post(f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-15-preview", 
                                 headers=headers, 
                                 json=payload)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        return response.json()['choices'][0]['message']['content']
    
    def get_embeddings(self, text):
        if not self.api_key or not self.endpoint or not self.model:
            raise ValueError("API key, endpoint, or model not found in environment variables")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        payload = {
            "input": text,
            "model": self.embeddings_model
        }

        response = requests.post(f"{self.endpoint}/openai/deployments/{self.embeddings_model}/embeddings?api-version=2024-02-15-preview", 
                                 headers=headers, 
                                 json=payload)

        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        return response.json()['data'][0]['embedding']

class GenericTools:
    def __init__(self):
        pass

    def clean_folder(self, folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    def create_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    def create_clean_folders(self, folders):
        for folder in folders:
            self.create_folder(folder)
            self.clean_folder(folder)

    def persist_scores_dataframe(self, scores, file_path):
        df = pd.DataFrame(scores)
        try:
            df.to_csv(file_path, index=False, encoding='latin-1')
            print(f"Persisted DataFrame to CSV at: {file_path}")
        except Exception as e:
            print(f"Failed to persist DataFrame to CSV. Reason: {e}")