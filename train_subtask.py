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
parser.add_argument("--warmup_epochs", type=int, default=1, help="number of warmup epochs (during which learning rate increases linearly from zero to set learning rate)")
parser.add_argument("--batch_size", type=int, default=1, help="eval batch size")
parser.add_argument("--shuffle", type=str, default=False, help="randomly shuffles data samples")
parser.add_argument("--mixed_precision", type=str, default=False, help="use mixed precision training")

parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")


#%%
args = parser.parse_args()
wandb.init(project=args.project)
wandb.config.update(args)
wandb.config.identifier = random_string

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.use_deterministic_algorithms(True)
torch.cuda.empty_cache()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

g = torch.Generator()
g.manual_seed(42)
set_seed(42)

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

step_global = 0

for epoch in range(args.epochs):
    losses = []
    mymodel.train()
    with tqdm.tqdm(train_loader) as progress_bar:
        for sample in progress_bar:
            step_global += args.batch_size
            #with torch.autograd.detect_anomaly():
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            #print(doc_ids)
            # --------- E2E Task  ------------#
            if args.mixed_precision:
                with autocast():
                    loss,events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text)
            else:
                loss,events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text)

            wandb.log({"eae_argex_loss": loss.item()}, step=step_global)

            losses.append(loss.item())
            progress_bar.set_postfix({"L":f"{sum(losses)/len(losses):.2f}"})

            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  
                loss.backward()
                nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
                optimizer.step()

            lr_scheduler.step()

            mymodel.zero_grad()
            del loss,events