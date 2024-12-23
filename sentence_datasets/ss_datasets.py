import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset  
import pandas as pd
import csv 

import os.path as osp

ROOT_DATA_DIR = "sentence_datasets/data/" 

class SentencePairDataset(Dataset):
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name 
        if dataset_name == "semeval24": 
            self.data = pd.read_csv(osp.join(ROOT_DATA_DIR, f"semeval24/{split}.csv"))  
        elif dataset_name == "mteb": 
            self.data = load_dataset("stsb_multi_mt", name="en", split=split)  
        elif dataset_name == "qqp": 
            with open(osp.join(ROOT_DATA_DIR, f"qqp/{split}.tsv"), encoding="utf-8") as f: 
                self.data = list(csv.DictReader(f, delimiter="\t")) 
        else: 
            raise NotImplementedError(f"does not support dataset {dataset_name}") 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.dataset_name == "semeval24":  
            row = self.data.iloc[idx]

            sentence1 = row["Text"].split("\n")[0]  # First line of Text
            sentence2 = row["Text"].split("\n")[1]  # Second line of Text
            similarity_score = row["Score"] 
            # assert similarity_score <= 1 and similarity_score >= 0 

        elif self.dataset_name == "mteb": 
            sentence1 = self.data[idx]["sentence1"]  
            sentence2 = self.data[idx]["sentence2"] 
            similarity_score = self.data[idx]["similarity_score"] 

        elif self.dataset_name == "qqp": 
            # return {
            #     "questions": [
            #         {"id": row["qid1"], "text": row["question1"]},
            #         {"id": row["qid2"], "text": row["question2"]},
            #     ],
            #     "is_duplicate": row["is_duplicate"] == "1",
            # } 
            data_row = self.data[idx] 
            sentence1 = data_row["question1"]  
            sentence2 = data_row["question2"] 
            similarity_score = int(data_row["is_duplicate"])  
        
        return {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "similarity_score": torch.tensor(similarity_score)  
        } 


if __name__ == "__main__":
    csv_file_path = "path/to/your/sentences.csv"
    
    dataset = SentencePairDataset("qqp", split="train") 

    sample = dataset[0]
    print(sample)