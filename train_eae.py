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
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import wandb
import argparse
import random
import string
from torch.cuda.amp import autocast, GradScaler

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()

parser.add_argument("--seed_model", type=int, default=123, help="random seed for model")
parser.add_argument("--seed_data", type=int, default=123, help="random seed for data")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train")
parser.add_argument("--train_set_size", type=str, default='small', help="size of the trainign set small | medium | large")


parser.add_argument("--batch_size_eval", type=int, default=4, help="eval batch size")
parser.add_argument("--batch_size_training", type=int, default=4, help="training batch size")
parser.add_argument("--warmup_epochs", type=int, default=1, help="warmup epochs")
parser.add_argument("--learning_rate", type=float, default=1e-6, help="learning rate")

parser.add_argument("--project", type=str, default="DocEE_eae", help="project name for wandb")
parser.add_argument("--full_task", type=str, default=False, help="True for full task, False for  eae subtask")
parser.add_argument("--soft_mention", type=str, default=False, help="method for mention detection")
parser.add_argument("--at_inference", type=str, default=False, help="use at labels for inference")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")




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
    parse_file(f"data/WikiEvents/preprocessed/train_{args.train_set_size}.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size_training,
    shuffle=False,
    collate_fn=collate_fn)
dev_loader = DataLoader(
    parse_file("data/WikiEvents/preprocessed/dev.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size_eval,
    shuffle=False,
    collate_fn=collate_fn)


# %%
# ------- Model and Optimizer Config -------#
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
optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate, eps=1e-6)
#scaler = GradScaler()
lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_epochs * len(train_loader)), int(args.num_epochs*len(train_loader)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%
# ---------- Train Loop -----------#
print("Training on {}".format(args.train_set_size))

#best_IDF_H_F1, best_IDF_C_F1, best_CLF_H_F1, best_CLF_C_F1 = 0.0, 0.0, 0.0, 0.0
best_compound_f1 = 0.0
step_global = -1
for i in tqdm.tqdm(range(args.num_epochs)):
    losses,argex_losses = [], []
    eae_event_list = []
    doc_id_list = []
    token_maps = []
    mymodel.train()
    with tqdm.tqdm(train_loader) as progress_bar:
        for sample in progress_bar:
            #with torch.autograd.detect_anomaly():
            
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            step_global += 1*len(doc_ids)

            _,argex_loss,loss,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=args.full_task)

            argex_losses.append(argex_loss.item())
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
            optimizer.step()

            mymodel.zero_grad()
            del loss#mention_loss,argex_loss,loss,eae_events


            progress_bar.set_postfix({"L":f"{sum(losses)/len(losses):.2f}"})
            wandb.log({"eae_loss": sum(losses)/len(losses)}, step=step_global)
            wandb.log({"eae_argex_loss": sum(argex_losses)/len(argex_losses)}, step=step_global)
            wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]}, step=step_global)

            
            
    mymodel.eval()
    print("Evaluation on dev")
    with tqdm.tqdm(dev_loader) as progress_bar:
        for sample in progress_bar:

            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample

            with torch.no_grad():
                _,_,_,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=args.full_task)
                for batch_i in range(input_ids.shape[0]):
                    doc_id_list.append(doc_ids[batch_i])
                    token_maps.append(token_map[batch_i])
                    try:
                        eae_event_list.append(eae_events[batch_i])
                    except:
                        eae_event_list.append([])
    eae_report = get_eval(eae_event_list,token_maps,doc_id_list)

    compound_f1 = eae_report["Identification"]["Head"]["F1"] + eae_report["Identification"]["Coref"]["F1"] + eae_report["Classification"]["Head"]["F1"] + eae_report["Classification"]["Coref"]["F1"]
    if compound_f1 > best_compound_f1:
        wandb.log({"eae_IDF_H_F1":eae_report["Identification"]["Head"]["F1"]})
        wandb.log({"eae_IDF_C_F1":eae_report["Identification"]["Coref"]["F1"]})
        wandb.log({"eae_CLF_H_F1":eae_report["Classification"]["Head"]["F1"]})
        wandb.log({"eae_CLF_C_F1":eae_report["Classification"]["Coref"]["F1"]})
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")
'''    
1. Habe nicht nur ein F1 Maß, sondern mehrere, passt das wenn es so einzeln geloggt wird?
So habe ich am Ende evtl. ein Modell gespeichert, dass einen bestimmten F1 Score maximiert der mich aber weniger interessiert.

    if eae_report["Identification"]["Head"]["F1"] > best_IDF_H_F1:
        wandb.log({"eae_IDF_H_F1":eae_report["Identification"]["Head"]["F1"] })
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")
    if eae_report["Identification"]["Coref"]["F1"] > best_IDF_C_F1:
        wandb.log({"eae_IDF_C_F1":eae_report["Identification"]["Coref"]["F1"] })
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")
    if eae_report["Classification"]["Head"]["F1"] > best_CLF_H_F1:
        wandb.log({"eae_CLF_H_F1":eae_report["Classification"]["Head"]["F1"] })
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")
    if eae_report["Classification"]["Coref"]["F1"] > best_CLF_C_F1:
        wandb.log({"eae_CLF_C_F1":eae_report["Classification"]["Coref"]["F1"] })
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")



2. Optimierung nach strengstem Maß: Head F1 Classification 
    if eae_report["Identification"]["Head"]["F1"] > best_IDF_H_F1:
        wandb.log({"eae_IDF_H_F1":eae_report["Identification"]["Head"]["F1"] })
        torch.save(mymodel.state_dict(), f"checkpoints/{random_string}.pt")
'''


