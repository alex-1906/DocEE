# %%
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data import parse_file, collate_fn
import tqdm
import json
from transformers.optimization import AdamW
import numpy as np
import pandas as pd
import torch.nn.functional as F
from src.losses import ATLoss
from src.util import process_long_input
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
from src.model import Encoder
from src.util import safe_div,compute_f1
from itertools import groupby
import wandb
import argparse
import string 
import random



# %%

#########################
## Model Configuration ##
#########################
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

df = pd.read_json('data/WikiEvents/train_eval.json')

with open("data/Ontology/roles.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/trigger_entity_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
dev_loader = DataLoader(
    parse_file("data/WikiEvents/train_docred_format_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn)

# %%

#%%
mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=False
                 )
optimizer = AdamW(mymodel.parameters(), lr=1e-5, eps=1e-6)
mymodel.load_state_dict(torch.load(f"checkpoints/my-test-project_MPFcphyWNy.pt"))

#%%
# ---------- Dev Eval ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymodel.to(device)
mymodel.train()

step_global = 0

losses = []
event_list = []
doc_id_list = []
with tqdm.tqdm(dev_loader) as progress_bar:
    for sample in progress_bar:
        step_global += 1
        token_ids, input_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        


        mymodel.train()
        loss ,triples, events = mymodel(token_ids.to(device), input_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        progress_bar.set_postfix({"L":f"{loss.item():.2f}"})
        wandb.log({"eae_loss": loss.item()})

        #insert events into dataframe
        for batch_i in range(token_ids.shape[0]):
            doc_id_list.append(doc_ids[batch_i])
            event_list.append(events[batch_i])
#%%
df['pred_events'] = pd.Series(event_list,index=doc_id_list)
df['pred_events'] = df['pred_events'].fillna("").apply(list)


#%%
idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0

for idx, row in df.iterrows(): 
    events = row['pred_events']
    gold_events = row['event_mentions']

    for ge,e in zip(gold_events,events):
        for g_arg in ge['arguments']:
            for arg in e['arguments']:
                #----- Head Matches -----
                if g_arg['entity_id'] == arg['entity_id']:
                    idf_h_matched += 1
                    idf_c_matched += 1
                    if g_arg['role'] == arg['role']:
                        clf_h_matched += 1
                        clf_c_matched += 1
                    
                #----- Coref Matches -----
                for coref in g_arg['corefs']:
                    if coref['entity_id'] == arg['entity_id']:
                        idf_c_matched += 1
                        if g_arg['role'] == arg['role']:
                            clf_c_matched += 1
            idf_gold += 1
        for arg in e['arguments']:
            idf_pred += 1
        clf_pred, clf_gold = idf_pred, idf_gold

        #----- Identification P,R,F1 -----
        idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
        idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

        #----- Classification P,R,F1 -----
        clf_h_p, clf_h_r, clf_h_f1 = compute_f1(idf_pred, idf_gold, clf_h_matched)
        clf_c_p, clf_c_r, clf_c_f1 = compute_f1(idf_pred, idf_gold, clf_c_matched)

#%%
print("***** Identification Report *****")
print(f"*Head* Matches: {idf_h_matched} Precision: {idf_h_p:.2f} Recall: {idf_h_r:.2f} F1-Score: {idf_h_f1:.2f}")
print(f"*Coref* Matches: {idf_c_matched} Precision: {idf_c_p:.2f} Recall: {idf_c_r:.2f} F1-Score: {idf_c_f1:.2f}")
print()
print("***** Classification Report *****")
print(f"*Head* Matches: {clf_h_matched} Precision: {clf_h_p:.2f} Recall: {clf_h_r:.2f} F1-Score: {clf_h_f1:.2f}")
print(f"*Coref* Matches: {clf_c_matched} Precision: {clf_c_p:.2f} Recall: {clf_c_r:.2f} F1-Score: {clf_c_f1:.2f}")
#%%

# %%
torch.save(mymodel.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")

#mymodel.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
# %%
