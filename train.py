#%%
import argparse
import pandas as pd
import json
import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          DistilBertConfig, RobertaConfig, XLMRobertaConfig)
from transformers.optimization import AdamW
from src.data import collate_fn, parse_file
from src.eval_util import get_eval, get_eval_by_id
from src.model import Encoder
from src.util import set_seed
import wandb
import argparse
import random
import string

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="test-e2e-gpu", help="project name for wandb")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")

parser.add_argument("--full_task", type=str, default=False, help="True for full task, False for  eae subtask")
parser.add_argument("--soft_mention", type=str, default=False, help="method for mention detection")
parser.add_argument("--at_inference", type=str, default=False, help="use at labels for inference")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")

parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="eval batch size")
parser.add_argument("--shuffle", type=str, default=False, help="randomly shuffles data samples")

parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")


#%%
args = parser.parse_args()
wandb.init(project=args.project)
wandb.config.update(args)
wandb.config.identifier = random_string

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

g = torch.Generator()
g.manual_seed(42)
set_seed(42)

# %%

#------- Model Configuration -------#

language_model = 'bert-base-uncased'
lm_config = AutoConfig.from_pretrained(
    language_model,
    num_labels=10,
)
lm_model = AutoModel.from_pretrained(
    language_model,
    from_tf=False,
    config=lm_config,
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


with open("data/Ontology/roles_shared.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/mention_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
train_loader = DataLoader(
    parse_file("data/WikiEvents/preprocessed/train_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    collate_fn=collate_fn)
dev_loader = DataLoader(
    parse_file("data/WikiEvents/preprocessed/dev.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=2,
    shuffle=args.shuffle,
    collate_fn=collate_fn)


# %%
mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=args.soft_mention,
                at_inference=args.at_inference,
                 )
optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate, eps=1e-7)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%
# ---------- Train Loop -----------#


for epoch in range(args.epochs):
    losses = []
    eae_event_list,e2e_event_list = [],[]
    doc_id_list = []
    token_maps = []
    mymodel.train()
    with tqdm.tqdm(train_loader) as progress_bar:
        for sample in progress_bar:
            #with torch.autograd.detect_anomaly():
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            print(doc_ids)
            # --------- E2E Task  ------------#
            mention_loss,argex_loss,loss,e2e_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)


            progress_bar.set_postfix({"L":f"{loss.item():.2f}"})
            wandb.log({"e2e_train_loss": loss.item()})
            wandb.log({"e2e_mention_loss": mention_loss.item()})
            wandb.log({"e2e_argex_loss": argex_loss.item()})

            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
            optimizer.step()

            #optimizer.zero_grad()
            mymodel.zero_grad()
            del mention_loss,argex_loss,loss,e2e_events
    mymodel.eval()
    with tqdm.tqdm(dev_loader) as progress_bar:
        for sample in progress_bar:
            
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            #print(doc_ids)
            # --------- E2E Task  ------------#
            with torch.no_grad():
                _,_,_,e2e_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)
                _,_,_,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)
                for batch_i in range(input_ids.shape[0]):
                    doc_id_list.append(doc_ids[batch_i])
                    token_maps.append(token_map[batch_i])
                    try:
                        e2e_event_list.append(e2e_events[batch_i])
                        eae_event_list.append(eae_events[batch_i])
                    except:
                        e2e_event_list.append([])
                        eae_event_list.append([])
    e2e_report = get_eval(e2e_event_list,token_maps,doc_id_list)
    eae_report = get_eval(eae_event_list,token_maps,doc_id_list)
    wandb.log({"e2e_IDF_C_F1":e2e_report["Identification"]["Coref"]["F1"] })
    wandb.log({"e2e_CLF_C_F1":e2e_report["Classification"]["Coref"]["F1"] })      
    wandb.log({"eae_IDF_C_F1":eae_report["Identification"]["Coref"]["F1"] })
    wandb.log({"eae_CLF_C_F1":eae_report["Classification"]["Coref"]["F1"] }) 
        
        

# %%
torch.save(mymodel.state_dict(), f"checkpoints/{args.checkpoint}.pt")

#%%
