from accelerate import Accelerator
from models.aggregators.salad import SinkhornAggregator
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sentence_datasets.ss_datasets import SentencePairDataset
from models.backbones.clip import CLIP 
from models.backbones.bert import Bert  
from models.backbones.roberta import Roberta  
import torch.nn.functional as F

sweep_configuration = {
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "val_loss"
    },
    "parameters": {
        "lr": {
            "values": [1e-4, 1e-5]
        },
        "model": {
            "values": ["clip", "ours"],  
        },  
        "num_clusters": {
            "values": [8, 16, 32], 
        }, 
        "cluster_dim": {
            "values": [32, 64, 128], 
        }, 
        "token_dim": {
            "values": [128, 256, 512],  
        }, 
        "num_trainable_blocks": {
            "values": [0, 1, 2, 3] 
        }, 
    }
}

WANDB = True
DATASET = "mteb" 
PROJECT = f"{DATASET}_anlp_tuning" 


class Pipeline(nn.Module):
    def __init__(self, model, aggregator=None, task="simil"):
        super().__init__()
        self.model = model
        self.task = task
        self.aggregator = aggregator
        # self.embedding_fn = {"simil" : self.give_embedding_simil}[self.task]
        self.embedding_fn = self.give_embedding_simil 
        self.loss_fn = {
            "simil" : self.simil_loss, 
            "simil2": self.simil_loss,
        }[self.task]
    
    def give_embedding_classification(self, data_dict):
        pass
    
    def give_embedding_simil(self, data_dict):
        sent1 = data_dict['sentence1']
        sent2 = data_dict['sentence2']
        sent1_tokens = self.model.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        sent2_tokens = self.model.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        for k in sent1_tokens.keys():
            sent1_tokens[k] = sent1_tokens[k].to(self.model.model.device)
            sent2_tokens[k] = sent2_tokens[k].to(self.model.model.device)
        
        output_sent1 = self.model(sent1_tokens) # (B, D)
        output_sent2 = self.model(sent2_tokens) # (B, D)
        
        if(self.aggregator is not None):
            output_sent1 = self.aggregator(output_sent1)
            output_sent2 = self.aggregator(output_sent2)
        else:
            output_sent1 = output_sent1[1]
            output_sent2 = output_sent2[1]
        
        output_sent1 = output_sent1 / output_sent1.norm(p=2, dim=-1, keepdim=True)
        output_sent2 = output_sent2 / output_sent2.norm(p=2, dim=-1, keepdim=True)
        pred_simil = torch.sum(output_sent1 * output_sent2, -1)
        return pred_simil
        
    def simil_loss(self, preds, data_dict):
        gt_simil_score = data_dict['similarity_score']
        # print(preds, gt_simil_score)
        loss = F.mse_loss(preds, gt_simil_score.float())
        return loss

    def bce_loss(self, preds, data_dict): 
        # gt_labels = data_dict['similarity_score'].to(torch.long)  
        loss = nn.BCELoss()(preds.float(), data_dict['similarity_score'].float()) 
        assert not torch.any(torch.isinf(preds)) and not torch.any(torch.isnan(preds)), f"{preds = }, {data_dict['similarity_score'] = }" 
        assert not torch.isnan(loss) and not torch.isinf(loss), f"{preds = }, {data_dict['similarity_score'] = }"  
        return loss 

    def ce_loss(self, preds, data_dict): 
        gt_labels = data_dict['similarity_score'].to(torch.long)
        loss = nn.CrossEntropyLoss()(preds, gt_labels) 
        return loss
    
    def __call__(self, data_dict):
        preds = self.embedding_fn(data_dict)
        loss = self.loss_fn(preds, data_dict)
        return loss  

# class Pipeline(nn.Module):
#     def __init__(self, model, task="simil"):
#         super().__init__()
#         self.model = model
#         self.task = task
#         self.embedding_fn = {"simil" : self.give_embedding_simil}[self.task]
#         self.loss_fn = {"simil" : self.simil_loss}[self.task]
    
#     def give_embedding_classification(self, data_dict):
#         pass
    
#     def give_embedding_simil(self, data_dict):
#         sent1 = data_dict['sentence1']
#         sent2 = data_dict['sentence2']
#         sent1_tokens = self.model.tokenizer(sent1, return_tensors="pt", padding=True, truncation=True)
#         sent2_tokens = self.model.tokenizer(sent2, return_tensors="pt", padding=True, truncation=True)

#         # sent1_tokens['input_ids'] = sent1_tokens['input_ids'].to(self.model.model.device)
#         # sent1_tokens['attention_mask'] = sent1_tokens['attention_mask'].to(self.model.model.device)
#         # sent2_tokens['input_ids'] = sent2_tokens['input_ids'].to(self.model.model.device)
#         # sent2_tokens['attention_mask'] = sent2_tokens['attention_mask'].to(self.model.model.device)

#         assert self.model.model.device != torch.device("cpu") 
#         for key in sent1_tokens.keys(): 
#             sent1_tokens[key] = sent1_tokens[key].to(self.model.model.device) 
#             sent2_tokens[key] = sent2_tokens[key].to(self.model.model.device) 
#         # sent1_tokens['input_ids'] = sent1_tokens['input_ids'].to(self.model.model.device)
#         # sent1_tokens['attention_mask'] = sent1_tokens['attention_mask'].to(self.model.model.device)
#         # sent2_tokens['input_ids'] = sent2_tokens['input_ids'].to(self.model.model.device)
#         # sent2_tokens['attention_mask'] = sent2_tokens['attention_mask'].to(self.model.model.device)
#         # assert sent1_tokens['input_ids'].device != torch.device("cpu") 
#         # assert sent2_tokens['input_ids'].device != torch.device("cpu") 

#         output_sent1 = self.model(sent1_tokens)[1] # (B, D)
#         output_sent2 = self.model(sent2_tokens)[1] # (B, D)
#         output_sent1 = output_sent1 / output_sent1.norm(p=2, dim=-1, keepdim=True)
#         output_sent2 = output_sent2 / output_sent2.norm(p=2, dim=-1, keepdim=True)
#         pred_simil = torch.sum(output_sent1 * output_sent2, -1)
#         return pred_simil

#     def simil_loss(self, preds, data_dict):
#         gt_simil_score = data_dict['similarity_score']
#         # print(preds, gt_simil_score)
#         loss = F.mse_loss(preds, gt_simil_score)
#         return loss
    
#     def __call__(self, data_dict):
#         preds = self.embedding_fn(data_dict)
#         loss = self.loss_fn(preds, data_dict)
#         return loss

def main(config, aggregator): 
    train_dataset = SentencePairDataset(config["dataset"], split="train") 
    val_dataset = SentencePairDataset(config["dataset"], split="dev") 
    bs = 128
    lr = config["lr"] 
    epochs = 30  
    # num_trainable_blocks = config["num_trainable_blocks"] 
    # model_name = config["model"] 
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True) 
    val_dataloader = DataLoader(val_dataset, batch_size=bs)  
    # if model_name == "clip": 
    #     model = CLIP(num_trainable_blocks=num_trainable_blocks, cache_dir="cache") 
    # elif model_name == "bert": 
    #     model = Bert(num_trainable_blocks=num_trainable_blocks, cache_dir="cache")   
    # elif model_name == "roberta": 
    #     model = Roberta(num_trainable_blocks=num_trainable_blocks, cache_dir="cache")   
    # model = model.to(accelerator.device) 

    pipeline = Pipeline(model, aggregator=aggregator, task=config["task"]).to(accelerator.device) 
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr)   
    pipeline, train_dataloader, val_dataloader, optimizer = accelerator.prepare(pipeline, train_dataloader, val_dataloader, optimizer) 

    for epoch in range(epochs):
        with tqdm(total=len(train_dataloader), desc=f"training on epoch {epoch+1}/{epochs}") as pbar: 
            wandb_log_data = {} 
            epoch_loss, epoch_val_loss = 0.0 , 0.0
            for batch in tqdm(train_dataloader):  
                optimizer.zero_grad()
                loss = pipeline(batch)
                epoch_loss += loss.detach() 
                accelerator.backward(loss) 
                optimizer.step()
                pbar.update(1) 
            epoch_loss = epoch_loss / len(train_dataloader)  
            
            with torch.no_grad():
                pbar.set_description(f"validating on epoch {epoch+1}/{epochs}") 
                for i, batch in enumerate(val_dataloader):
                    loss = pipeline(batch)
                    epoch_val_loss += loss.detach()
                
                epoch_val_loss = epoch_val_loss / len(val_dataloader) 
                
            gathered_epoch_loss = accelerator.gather(epoch_loss.unsqueeze(0)) 
            gathered_epoch_val_loss = accelerator.gather(epoch_val_loss.unsqueeze(0)) 
            
            if accelerator.is_main_process and WANDB: 
                wandb_log_data['train_loss'] = torch.mean(gathered_epoch_loss, 0).item()
                wandb_log_data['val_loss'] = torch.mean(gathered_epoch_val_loss, 0).item()
                wandb.log(wandb_log_data)



if __name__ == "__main__": 
    accelerator = Accelerator()
    # iterate over all the possible combinations of hyperparameters 
    for model_name in ["bert", "clip", "roberta"]:  
        for num_trainable_blocks in [1]: 
            if model_name == "clip": 
                model = CLIP(num_trainable_blocks=num_trainable_blocks, cache_dir="cache") 
            elif model_name == "bert": 
                model = Bert(num_trainable_blocks=num_trainable_blocks, cache_dir="cache")   
            elif model_name == "roberta": 
                model = Roberta(num_trainable_blocks=num_trainable_blocks, cache_dir="cache")   
            model = model.to(accelerator.device) 
            a = SinkhornAggregator(num_channels=model.model.config.hidden_size)
            for aggregator in [a, None]: 
                is_aggregator = aggregator is not None  
                if aggregator is not None: 
                    for lr in [1e-4]: 
                        for num_clusters in [16]: 
                            for cluster_dim in [128]: 
                                for token_dim in [128]: 
                                    task = "simil" 
                                    config = {
                                        "lr": lr, 
                                        "task": task,  
                                        "dataset": DATASET, 
                                        "model_name": model_name, 
                                        "aggregator": is_aggregator,  
                                        "num_clusters": num_clusters, 
                                        "cluster_dim": cluster_dim, 
                                        "token_dim": token_dim, 
                                        "num_trainable_blocks": num_trainable_blocks 
                                    }
                                    if accelerator.is_main_process: 
                                        run_name = f"{model_name}_{is_aggregator}_{lr}_{num_clusters}_{cluster_dim}_{token_dim}_{num_trainable_blocks}" 
                                        wandb.init(project=PROJECT, name=run_name, config=config) 
                                        # wandb.config.lr = lr 
                                        # wandb.config.model = model 
                                        # wandb.config.num_clusters = num_clusters 
                                        # wandb.config.cluster_dim = cluster_dim 
                                        # wandb.config.token_dim = token_dim 
                                        # wandb.config.num_trainable_blocks = num_trainable_blocks
                                    main(config, aggregator) 
                                    if accelerator.is_main_process: 
                                        wandb.finish() 
                else: 
                    for lr in [1e-4]: 
                        config = {
                            "lr": lr, 
                            "model_name": model_name, 
                            "aggregator": is_aggregator,  
                            "task": task,  
                            "dataset": DATASET, 
                            "num_trainable_blocks": num_trainable_blocks 
                        }

                        if accelerator.is_main_process and WANDB: 
                            run_name = f"{model_name}_{is_aggregator}_{lr}_{num_trainable_blocks}" 
                            wandb.init(project=PROJECT, name=run_name, config=config) 

                        main(config, aggregator) 
                        if accelerator.is_main_process and WANDB: 
                            wandb.finish() 