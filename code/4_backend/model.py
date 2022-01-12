from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
from torch import nn

FINAL_MODEL_PATH="models/final/roberta-base" # the directory stored the fine-tuned model's config and weights 
MAX_LENGTH = 256 #the length of the input sequence to the model
BINS = [float('inf'), 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, float('-inf')] # map the raw score to readability level from 1 to 12(easy to hard)

class Readabilitymodel():
    def __init__(self,model_name,model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_dir, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=self.config)
    
    def predict(self,excerpt):
        embeddings=self.tokenizer(excerpt, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        self.model.eval()
        inputs = {"input_ids": embeddings['input_ids'],"attention_mask": embeddings['attention_mask']}
        with torch.no_grad():
            outputs = self.model(**inputs).logits
            score=outputs.view(-1).item()
        level=np.digitize(score,BINS)
        return score,level