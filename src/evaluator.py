import json, os, pandas as pd
import numpy as np
from dotenv import load_dotenv
from string import Template
from scipy import stats
from helper import AzureOpenAI, GenericTools
import prompts as prt

load_dotenv()

class TranscriptionEvaluator:
    def __init__(self, folder_groundtruth):
        self.folder_groundtruth = folder_groundtruth
        self.azure_openai = AzureOpenAI()

    def _remove_groundtruth_prefix(self, filename):
        return filename.replace('groundtruth_transcription_', '')

    def calculate_llm_score(self, folder_transcriptions):
        scores = []
        for file in os.listdir(self.folder_groundtruth):
            if file.endswith(".txt"):
                groundtruth = self._read_file(self.folder_groundtruth, file)
                transcription_files = self._get_transcription_files(self._remove_groundtruth_prefix(file), 
                                                                    folder_transcriptions)

                for transcription_file in transcription_files:
                    transcription = self._read_file(folder_transcriptions, transcription_file)
                    result = self._get_similarity_score(groundtruth, transcription)
                    scores.append({
                        "filename": transcription_file,
                        "similarity-score": result["similarity-score"],
                        "reason": result["reason"]
                    })
        return scores

    def generate_embeddings(self, folder, suffix):
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                transcription = self._read_file(folder, file)
                result = self.azure_openai.get_embeddings(transcription)
                self._save_embeddings(result, folder, suffix, file)

    def calculate_embeddings_similarity_score(self, files_groundtruth, folder_path):
        scores = []
        for file in files_groundtruth:            
            groundtruth = np.load(f"{folder_path}/{file}", allow_pickle=True)
            ai_transcription_files = self._get_ai_transcription_files(self._remove_groundtruth_prefix(file), 
                                                                      folder_path)

            for ai_file in ai_transcription_files:
                ai_transcription = np.load(f"{folder_path}/{ai_file}", allow_pickle=True)
                similarity_score = self._cosine_similarity(groundtruth, ai_transcription)
                ks_test = self._kstest_similarity(groundtruth, ai_transcription)
                scores.append({
                    "filename": ai_file,
                    "similarity-score": float(similarity_score),
                    "ks-test-pvalue": float(ks_test.pvalue),
                    "ks-test-stats": float(ks_test.statistic)
                })
        return scores

    def _read_file(self, folder, file):
        with open(f"{folder}/{file}", "r") as f:
            return f.read()

    def _get_transcription_files(self, file, folder_transcriptions):
        return [f_adj for f_adj in os.listdir(folder_transcriptions) 
                if file.replace('.txt', '') in f_adj]

    def _get_similarity_score(self, groundtruth, adjusted_transcription):
        user_prompt = Template(prt.user_prompt_evaluation_similarity_score)
        result = self.azure_openai.send_llm_request(prt.system_prompt_evaluation_similarity_score,
                                                    user_prompt.substitute(groundtruth=groundtruth, 
                                                                           adjusted_transcription=adjusted_transcription))
        return json.loads(result)

    def _save_embeddings(self, result, folder, suffix, file):
        np.array(result).dump(f"{folder.replace('/' + suffix, '')}/embeddings/{suffix}-{file.replace('.txt', '.npy')}")

    def _get_ai_transcription_files(self, file, folder_path):
        return [f_adj for f_adj in os.listdir(folder_path) 
                if file.replace('.npy', '').replace('groundtruth-', '') in f_adj 
                and 'groundtruth' not in f_adj]

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _kstest_similarity(self, a, b):
        return stats.kstest(a, b)
    
    def write_scores(self, scores, file_path='evaluations/scores.csv'):
        generic_tools = GenericTools()
        generic_tools.persist_scores_dataframe(scores, file_path)
    
    def evaluate_transcriptions(self, folder):
        evaluation = []
        # Loop through transcriptions in the folder
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename)) as f:
                    transcription = f.read()
                    response = AzureOpenAI().send_llm_request(prt.system_prompt_evaluation, 
                                                              Template(prt.user_prompt_evaluation).substitute(
                                                                  transcription=transcription))
                    
                    # Append the evaluation to the list of evaluations
                    evaluation.append({
                        "filename": filename,
                        "evaluation": json.loads(response)
                    })
        
        return evaluation
    
    def analyze_evaluation(self, evaluation):
        # Create a DataFrame from the evaluation
        df = pd.DataFrame(evaluation)

        # Normalize the nested JSON in the "evaluation" column
        evaluation_normalized = pd.json_normalize(df['evaluation'])

        # Concatenate the normalized data with the original DataFrame
        df = pd.concat([df.drop(['evaluation'], axis=1), evaluation_normalized], axis=1)

        # Rename the columns removing the evaluation prefix
        df.columns = df.columns.str.replace('evaluation.', '')

        # Split the filename into parts using _ as the separator
        df[['API', 'segment', 'filename']] = df['filename'].str.split('_', expand=True)

        # Sort the DataFrame by the filename and API columns
        df = df.sort_values(by=['filename', 'API'])

        return df

if __name__ == '__main__':
    generic_tools = GenericTools()
    generic_tools.create_clean_folders(['transcriptions/evaluations', 'transcriptions/embeddings'])

    evaluator = TranscriptionEvaluator('transcriptions/groundtruth')

    ### LLM Similarity Score ###
    # Calculate the similarity score (using LLM) between the groundtruth and raw transcriptions
    scores_raw = evaluator.calculate_llm_score("transcriptions/raw")

    # Calculate the similarity score (using LLM) between the groundtruth and adjusted transcriptions
    scores_adjusted = evaluator.calculate_llm_score("transcriptions/adjusted")
    
    # Persist the similarity scores
    evaluator.write_scores(scores_raw, 'transcriptions/evaluations/scores-llm-raw.csv')
    evaluator.write_scores(scores_adjusted, 'transcriptions/evaluations/scores-llm-adjusted.csv')
    ###################################

    ### Embeddings Similarity Score ###
    evaluator.generate_embeddings('transcriptions/groundtruth', 'groundtruth')
    evaluator.generate_embeddings('transcriptions/raw', 'raw')
    evaluator.generate_embeddings('transcriptions/adjusted', 'adjusted')

    # Calculate the similarity score (using embeddings) between the groundtruth and raw transcriptions
    embeddings_files = [f for f in os.listdir("transcriptions/embeddings") 
                        if f.endswith(".npy") and "groundtruth" in f]
    scores_embeddings = evaluator.calculate_embeddings_similarity_score(embeddings_files, 
                                                                        'transcriptions/embeddings')

    # Persist the embeddings similarity scores
    evaluator.write_scores(scores_embeddings, 'transcriptions/evaluations/scores-embeddings.csv')
    ###################################

    ### Evaluation Analysis ###
    evaluation_adjusted = evaluator.evaluate_transcriptions("transcriptions/adjusted")
    evaluation_groundtruth = evaluator.evaluate_transcriptions("transcriptions/groundtruth")
    evaluation = evaluation_adjusted + evaluation_groundtruth

    df = evaluator.analyze_evaluation(evaluation)
    df.to_csv('transcriptions/evaluations/evaluation.csv', index=False, encoding='latin-1')

    print("Evaluation completed successfully")