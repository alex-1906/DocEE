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
from src.model import Encoder
from src.util import set_seed
import wandb
import argparse
import random
import string


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
# %%


shared_roles = False

if shared_roles:
    with open("data/Ontology/roles_shared.json") as f:
        relation_types = json.load(f)
else:
    with open("data/Ontology/roles_unique.json") as f:
        relation_types = json.load(f)
with open("data/Ontology/mention_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
len(relation_types)
#%%


train_file,dev_file,test_file="train.json","dev.json","test.json"
train_loader = DataLoader(
    parse_file(f"data/WikiEvents/preprocessed/train_medium.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn)

#%%
mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=False,
                at_inference=False,
                 )

#%%

sample = next(iter(train_loader))

input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample

mention_loss,argex_loss,loss,_ = mymodel(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)

print( "test")

#%%
sample = next(iter(train_loader))

input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample

#%%
input_ids
# %%
for sample in train_loader:
    input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
    for r in relation_labels:
        for l in r.items():
            if(l[1]>=60):
                print(l)
   
# %%
