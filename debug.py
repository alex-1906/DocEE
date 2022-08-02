#%%
import argparse


import pandas as pd
import json
import os
import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          DistilBertConfig, RobertaConfig, XLMRobertaConfig)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.data import collate_fn, parse_file
from src.eval_util import get_eval, get_eval_by_id
from src.subtask import Encoder


epochs = 1
mixed_precision = False

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
    parse_file("data/WikiEvents/preprocessed/train_medium.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn)



# %%
mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                feasible_roles=feasible_roles,
                 )
optimizer = AdamW(mymodel.parameters(), lr=1e-6, eps=1e-7)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(1 * len(train_loader)), int(epochs*len(train_loader)))
if mixed_precision:
    scaler = GradScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%
# ---------- Train Loop -----------#

losses = []
eae_event_list = []
doc_id_list = []
token_maps = []
mymodel.train()

#%%
input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = next(iter(train_loader))

#%%
loss,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text)
loss.backward()
#%%
for batch_i in range(input_ids.shape[0]):
    try:
        eae_event_list.append(eae_events[batch_i])
    except:
        eae_event_list.append([ ])
    doc_id_list.append(doc_ids[batch_i])
    token_maps.append(token_map[batch_i])


print(len(eae_event_list))
print(len(doc_id_list))
print(len(token_maps))
#%%
#eae_report = get_eval(eae_event_list,token_maps,doc_id_list)


# %%
eae_events
# %%
