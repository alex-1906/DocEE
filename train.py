#%%
from datetime import datetime
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
from src.eval_util import get_eval
from src.debug_model import Encoder
from src.util import set_seed
import wandb
import argparse
import random
import string

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()

parser.add_argument("--random_seed", type=int, default=123, help="random seed")
parser.add_argument("--project", type=str, default="full_working", help="project name for wandb")

parser.add_argument("--train_file", type=str, default="train.json", help="train file")
parser.add_argument("--dev_file", type=str, default="dev.json", help="dev file")
parser.add_argument("--test_file", type=str, default="test.json", help="test file")

parser.add_argument("--full_task", type=str, default="False", help="True for full task, False for  eae subtask")
parser.add_argument("--shared_roles", type=str, default="True", help="Shared Role Types")
parser.add_argument("--coref", type=str, default=False, help="Use coref mentions for embedding")

parser.add_argument("--soft_mention", type=str, default=False, help="method for mention detection")
parser.add_argument("--at_inference", type=str, default=False, help="use at labels for inference")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")

parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument("--warmup_epochs", type=int, default=1, help="number of warmup epochs (during which learning rate increases linearly from zero to set learning rate)")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--shuffle", type=str, default=False, help="randomly shuffles data samples")
parser.add_argument("--mixed_precision", type=str, default=False, help="use mixed precision training")

parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--num_trigger_prototypes", type=float, default=1, help="number of prototypes for trigger extraction" ) 
parser.add_argument("--loss_ratio", type=float, default=1, help="ratio of mention and argex loss")
parser.add_argument("--max_doc_len", type=float, default=1500, help="maximum length of a document")



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
g.manual_seed(args.random_seed)
set_seed(args.random_seed)

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

if args.shared_roles == 'True':
    shared_roles = True
    with open("data/Ontology/roles_shared.json") as f:
        relation_types = json.load(f)
else:
    shared_roles = False
    with open("data/Ontology/roles_unique.json") as f:
        relation_types = json.load(f)
with open("data/Ontology/mention_types.json") as f:
    mention_types = json.load(f)

if args.num_trigger_prototypes == 2:
    with open("data/Ontology/mention_types_large.json") as f:
        mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9

if args.coref == 'True':
    args.train_file = "coref/"+args.train_file
    args.dev_file = "coref/"+args.dev_file
    args.test_file = "coref/"+args.test_file

#bug with parser
if args.full_task == "True":
    full_task = True
else:
    full_task = False




print("\n------------------Configurations--------------------\n")

print(f"Full task:            {full_task}")
print(f"Shared roles:         {shared_roles}")
print(f"Loss ratio:           {args.loss_ratio}")
print(f"Train file:           {args.train_file}")
print(f"Dev file:             {args.dev_file}")
print(f"Test file:            {args.test_file}")

print("\n----------------------------------------------------\n")

train_loader = DataLoader(
    parse_file(f"data/WikiEvents/preprocessed/{args.train_file}",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    collate_fn=collate_fn)
dev_loader = DataLoader(
    parse_file(f"data/WikiEvents/preprocessed/{args.dev_file}",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    collate_fn=collate_fn)
test_loader = DataLoader(
    parse_file(f"data/WikiEvents/preprocessed/{args.test_file}",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
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
                num_trigger_prototypes = args.num_trigger_prototypes
                 )
num_params = sum(param.numel() for param in mymodel.parameters())
wandb.log({"complexity":num_params})
optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate, eps=1e-7)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_epochs * len(train_loader)), int(args.epochs*len(train_loader)))
if args.mixed_precision:
    scaler = GradScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%


step_global = 0
best_compound_f1 = 0.0
for epoch in tqdm.tqdm(range(args.epochs)):

    if epoch <= 40:
        loss_ratio = 1
    else:
        loss_ratio = args.loss_ratio
        

    # ---------- Train Loop -----------#
    losses = []
    event_list = []
    doc_id_list = []
    token_maps = []
    mymodel.train()
    with tqdm.tqdm(train_loader) as progress_bar:
        for sample in progress_bar:

            step_global += args.batch_size
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample

            #-------- Skip long docs ---------
            if input_ids.size(1) > args.max_doc_len:
                print(f"--------- long doc {input_ids.size(1)} skipped ---------")
                continue


            t_start = datetime.now()
            mention_loss,argex_loss,loss,_ = mymodel(doc_ids,input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=full_task)
            t_end = datetime.now()
            t_time = (t_end - t_start).total_seconds()
            
            wandb.log({"mention_loss": mention_loss.item()}, step=step_global) 
            wandb.log({"argex_loss": argex_loss.item()}, step=step_global) 
            wandb.log({"loss": loss.item()}, step=step_global)
            wandb.log({"training_time":t_time}, step=step_global)

            losses.append(loss.item())
            progress_bar.set_postfix({"L":f"{sum(losses)/len(losses):.2f}"})
            if args.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  
                loss = loss_ratio*mention_loss + (1-loss_ratio)*argex_loss
                loss.backward()
                nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
                optimizer.step()

            lr_scheduler.step()
            mymodel.zero_grad()
            del mention_loss,argex_loss,loss,_


# ----------- Evaluation on Dev --------------            
    mymodel.eval()
    with tqdm.tqdm(dev_loader) as progress_bar:
        for sample in progress_bar:
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
           
            #-------- Skip long docs ---------
            if input_ids.size(1) > args.max_doc_len:
                print(f"--------- long doc {input_ids.size(1)} skipped ---------")
                continue
            
            i_start = datetime.now()
            with torch.no_grad():
                _,_,_,events = mymodel(doc_ids,input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=full_task)
                i_end = datetime.now()
                i_time = (i_end - i_start).total_seconds()
                wandb.log({"inference_time":i_time}, step=step_global)

                for batch_i in range(input_ids.shape[0]):
                    doc_id_list.append(doc_ids[batch_i])
                    token_maps.append(token_map[batch_i])

                    # Block wird nicht mehr gebraucht
                    try:
                        event_list.append(events[batch_i])
                    except:
                        print("----------in except block---------")
                        event_list.append([])

    report = get_eval(event_list,token_maps,doc_id_list)
    print(report)
    if full_task:
        compound_f1 = (report["Identification"]["Head"]["F1"] + report["Identification"]["Coref"]["F1"] + report["Classification"]["Head"]["F1"] + report["Classification"]["Coref"]["F1"] + report["Trigger"]["Identification"]["F1"] + report["Trigger"]["Classification"]["F1"])/6
    else:
        compound_f1 = (report["Identification"]["Head"]["F1"] + report["Identification"]["Coref"]["F1"] + report["Classification"]["Head"]["F1"] + report["Classification"]["Coref"]["F1"])/4

    wandb.log({"Dev_Compound_F1":compound_f1 }, step=step_global)      
    wandb.log({"Dev_Identification_Head_F1":report["Identification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"Dev_Identification_Coref_F1":report["Identification"]["Coref"]["F1"] }, step=step_global)
    wandb.log({"Dev_Classification_Head_F1":report["Classification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"Dev_Classification_Coref_F1":report["Classification"]["Coref"]["F1"] }, step=step_global)  
    wandb.log({"Dev_Trigger_Identification_F1":report["Trigger"]["Identification"]["F1"] }, step=step_global)
    wandb.log({"Dev_Trigger_Classification_F1":report["Trigger"]["Classification"]["F1"] }, step=step_global)   

    if compound_f1 > best_compound_f1:
        torch.save(mymodel.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")

# --------- Evaluation on Test  ------------#
try:
    mymodel.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
except:
    print("Could not load checkpoint")
    pass
mymodel.eval()
event_list = []
doc_id_list = []
token_maps = []
with tqdm.tqdm(test_loader) as progress_bar:
    for sample in progress_bar:
        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        #-------- Skip long docs ---------
        if input_ids.size(1) > args.max_doc_len:
                print(f"--------- long doc {input_ids.size(1)} skipped ---------")
                continue
        with torch.no_grad():
            _,_,_,events = mymodel(doc_ids,input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=full_task)
            
            for batch_i in range(input_ids.shape[0]):
                doc_id_list.append(doc_ids[batch_i])
                token_maps.append(token_map[batch_i])
                try:
                    event_list.append(events[batch_i])
                    #Block wird nicht gebraucht
                except:
                    print("in exception block")
                    event_list.append([])

report = get_eval(event_list,token_maps,doc_id_list)
print(report)
if full_task:
    compound_f1 = (report["Identification"]["Head"]["F1"] + report["Identification"]["Coref"]["F1"] + report["Classification"]["Head"]["F1"] + report["Classification"]["Coref"]["F1"] + report["Trigger"]["Identification"]["F1"] + report["Trigger"]["Classification"]["F1"])/6
else:
    compound_f1 = (report["Identification"]["Head"]["F1"] + report["Identification"]["Coref"]["F1"] + report["Classification"]["Head"]["F1"] + report["Classification"]["Coref"]["F1"])/4

wandb.log({"Test_Compound_F1":compound_f1 }, step=step_global)      
wandb.log({"Test_Identification_Head_F1":report["Identification"]["Head"]["F1"] }, step=step_global)
wandb.log({"Test_Identification_Coref_F1":report["Identification"]["Coref"]["F1"] }, step=step_global)
wandb.log({"Test_Classification_Head_F1":report["Classification"]["Head"]["F1"] }, step=step_global)
wandb.log({"Test_Classification_Coref_F1":report["Classification"]["Coref"]["F1"] }, step=step_global)  
wandb.log({"Test_Trigger_Identification_F1":report["Trigger"]["Identification"]["F1"] }, step=step_global)
wandb.log({"Test_Trigger_Classification_F1":report["Trigger"]["Classification"]["F1"] }, step=step_global)  

#%%