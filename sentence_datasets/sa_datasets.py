import pandas as pd
import torch
from torch.utils.data import Dataset

# Custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, file_path):
        dataframe = pd.read_csv(file_path)
        dataframe['Sentiment'] = dataframe['Sentiment'].str.strip()
        self.data = dataframe
        self.texts = dataframe['Text'].values
        self.sentiments = dataframe['Sentiment'].values
        self.sentiment_to_idx = {'Positive': 1, 'Negative': 0, 'Neutral': 2}  # Mapping sentiments to integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        sentiment = self.sentiment_to_idx[self.sentiments[idx]]
        return {
            "sentence": sentence,
            "sentiment": torch.tensor(sentiment, dtype=torch.long)
        }
    
# Load the CSV file
file_path = "sentimentdataset.csv"  # Update this with the actual file path
dataset = SentimentDataset(file_path) 
print(dataset[0])