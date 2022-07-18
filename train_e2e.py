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
from src.e2emodel import Encoder
from src.util import safe_div,compute_f1
from itertools import groupby
import wandb
import argparse
import string 
import random

wandb.init(project='my-first-project')

wandb.config = {
  "learning_rate": 1e-6,
  "epochs": 1,
  "batch_size": 4,
  "shuffle": False
}

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

df = pd.read_json('data/Dump/train_eval.json')

with open("data/Ontology/roles.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/trigger_entity_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
dev_loader = DataLoader(
    parse_file("data/WikiEvents/DocRed_Format/train_small.json",
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
# ---------- Train Loop ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymodel.to(device)
mymodel.train()

step_global = 0

losses = []
event_list = []
doc_id_list = []
token_maps = []
with tqdm.tqdm(dev_loader) as progress_bar:
    for sample in progress_bar:
        step_global += 1
        token_ids, input_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        


        loss ,triples, events = mymodel(token_ids.to(device), input_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text)
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
            token_maps.append(token_map[batch_i])
#%%
# ---------- Eval ------------

from src.eval_util import get_eval

report = get_eval(event_list,token_maps,doc_id_list)
report
# %%

# %%
