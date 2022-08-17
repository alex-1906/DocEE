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

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=123, help="random seed")
parser.add_argument("--project", type=str, default="GPU", help="project name for wandb")
parser.add_argument("--train_file", type=str, default="train.json", help="train file")
parser.add_argument("--dev_file", type=str, default="dev.json", help="dev file")
parser.add_argument("--test_file", type=str, default="test.json", help="test file")
#parser.add_argument("--overfit_test", type=str, default=False, help="True for full task, False for  eae subtask")

parser.add_argument("--full_task", type=str, default=False, help="True for full task, False for  eae subtask")
parser.add_argument("--shared_roles", type=str, default=False, help="Shared Role Types")
parser.add_argument("--coref", type=str, default=False, help="Use coref mentions for embedding")

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
    print(f"\nshared roles\n")
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

# if args.overfit_test:
#     if args.coref:
#         train_file,dev_file,test_file="coref/train_medium.json","coref/train_medium.json","coref/train_medium.json"
#     else:
#         train_file,dev_file,test_file="train_medium.json","train_medium.json","train_medium.json"
# else:
#     if args.coref:
#         train_file,dev_file,test_file="coref/train_medium.json","coref/dev.json","coref/test.json"
#     else: 
#         train_file,dev_file,test_file="train_medium.json","dev.json","test.json"
if args.coref == 'True':
    args.train_file = "coref/"+args.train_file
    args.dev_file = "coref/"+args.dev_file
    args.test_file = "coref/"+args.test_file
print(f"\nTraining on {args.train_file}; Evaluating on {args.dev_file}\n; Testing on {args.test_file}")
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
    batch_size=2,
    shuffle=args.shuffle,
    collate_fn=collate_fn)
test_loader = DataLoader(
    parse_file(f"data/WikiEvents/preprocessed/{args.test_file}",
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
lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_epochs * len(train_loader)), int(args.epochs*len(train_loader)))
if args.mixed_precision:
    scaler = GradScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%
# ---------- Train Loop -----------#

step_global = 0
eae_best_compound_f1, e2e_best_compound_f1  = 0.0,0.0
for epoch in tqdm.tqdm(range(args.epochs)):
    losses = []
    eae_event_list,e2e_event_list = [],[]
    doc_id_list = []
    token_maps = []
    mymodel.train()
    with tqdm.tqdm(train_loader) as progress_bar:
        for sample in progress_bar:
            step_global += args.batch_size
            #with torch.autograd.detect_anomaly():
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            # --------- E2E Task  ------------#
            if args.mixed_precision:
                with autocast():
                    mention_loss,argex_loss,loss,_ = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=args.full_task)
            else:
                mention_loss,argex_loss,loss,_ = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=args.full_task)


            wandb.log({"mention_loss": mention_loss.item()}, step=step_global) 
            wandb.log({"argex_loss": argex_loss.item()}, step=step_global) 
            wandb.log({"loss": loss.item()}, step=step_global)

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

            #optimizer.zero_grad()
            mymodel.zero_grad()
            del mention_loss,argex_loss,loss,_
    mymodel.eval()
    with tqdm.tqdm(dev_loader) as progress_bar:
        for sample in progress_bar:
            
            input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
            # --------- E2E Task  ------------#
            with torch.no_grad():
                if args.full_task:
                    _,_,_,e2e_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=True)
                _,_,_,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)
                for batch_i in range(input_ids.shape[0]):
                    doc_id_list.append(doc_ids[batch_i])
                    token_maps.append(token_map[batch_i])
                    try:
                        if args.full_task:
                            e2e_event_list.append(e2e_events[batch_i])
                        eae_event_list.append(eae_events[batch_i])
                    except:
                        e2e_event_list.append([])
                        eae_event_list.append([])



    if args.full_task:
        e2e_report = get_eval(e2e_event_list,token_maps,doc_id_list)
        e2e_compound_f1 = (e2e_report["Identification"]["Head"]["F1"] + e2e_report["Identification"]["Coref"]["F1"] + e2e_report["Classification"]["Head"]["F1"] + e2e_report["Classification"]["Coref"]["F1"])/4
    eae_report = get_eval(eae_event_list,token_maps,doc_id_list)
    eae_compound_f1 = (eae_report["Identification"]["Head"]["F1"] + eae_report["Identification"]["Coref"]["F1"] + eae_report["Classification"]["Head"]["F1"] + eae_report["Classification"]["Coref"]["F1"])/4
    #to check if offset F1s match with ID F1s
    eae_report = get_eval_by_id(eae_event_list,token_maps,doc_id_list)
    wandb.log({"eae_Compound_F1_id":eae_compound_f1 }, step=step_global) 
    if args.full_task:
        wandb.log({"e2e_Compound_F1":e2e_compound_f1 }, step=step_global)  
        wandb.log({"e2e_Identification_Head_F1":e2e_report["Identification"]["Head"]["F1"] }, step=step_global)
        wandb.log({"e2e_Identification_Coref_F1":e2e_report["Identification"]["Coref"]["F1"] }, step=step_global)
        wandb.log({"e2e_Classification_Head_F1":e2e_report["Classification"]["Head"]["F1"] }, step=step_global)
        wandb.log({"e2e_Classification_Coref_F1":e2e_report["Classification"]["Coref"]["F1"] }, step=step_global)  

    wandb.log({"eae_Compound_F1":eae_compound_f1 }, step=step_global)      
    wandb.log({"eae_Identification_Head_F1":eae_report["Identification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"eae_Identification_Coref_F1":eae_report["Identification"]["Coref"]["F1"] }, step=step_global)
    wandb.log({"eae_Classification_Head_F1":eae_report["Classification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"eae_Classification_Coref_F1":eae_report["Classification"]["Coref"]["F1"] }, step=step_global)  
    if eae_compound_f1 > eae_best_compound_f1:
        torch.save(mymodel.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")

# --------- Evaluation on Test  ------------#
print("---- TEST EVAL -----")
try:
    mymodel.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
except:
    pass
mymodel.eval()
eae_event_list,e2e_event_list = [],[]
doc_id_list = []
token_maps = []
with tqdm.tqdm(test_loader) as progress_bar:
    for sample in progress_bar:
        
        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        # --------- E2E Task  ------------#
        with torch.no_grad():
            if args.full_task:
                _,_,_,e2e_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=True)
            _,_,_,eae_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)
            for batch_i in range(input_ids.shape[0]):
                doc_id_list.append(doc_ids[batch_i])
                token_maps.append(token_map[batch_i])
                try:
                    if args.full_task:
                        e2e_event_list.append(e2e_events[batch_i])
                    eae_event_list.append(eae_events[batch_i])
                except:
                    e2e_event_list.append([])
                    eae_event_list.append([])

if args.full_task:
    e2e_report = get_eval(e2e_event_list,token_maps,doc_id_list)
    e2e_compound_f1 = (e2e_report["Identification"]["Head"]["F1"] + e2e_report["Identification"]["Coref"]["F1"] + e2e_report["Classification"]["Head"]["F1"] + e2e_report["Classification"]["Coref"]["F1"])/4
eae_report = get_eval(eae_event_list,token_maps,doc_id_list)
eae_compound_f1 = (eae_report["Identification"]["Head"]["F1"] + eae_report["Identification"]["Coref"]["F1"] + eae_report["Classification"]["Head"]["F1"] + eae_report["Classification"]["Coref"]["F1"])/4
#to check if offset F1s match with ID F1s
eae_report = get_eval_by_id(eae_event_list,token_maps,doc_id_list)
wandb.log({"Test_eae_Compound_F1_id":eae_compound_f1 }, step=step_global) 
if args.full_task:
    wandb.log({"Test_e2e_Compound_F1":e2e_compound_f1 }, step=step_global)  
    wandb.log({"Test_e2e_Identification_Head_F1":e2e_report["Identification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"Test_e2e_Identification_Coref_F1":e2e_report["Identification"]["Coref"]["F1"] }, step=step_global)
    wandb.log({"Test_e2e_Classification_Head_F1":e2e_report["Classification"]["Head"]["F1"] }, step=step_global)
    wandb.log({"Test_e2e_Classification_Coref_F1":e2e_report["Classification"]["Coref"]["F1"] }, step=step_global)  

wandb.log({"Test_eae_Compound_F1":eae_compound_f1 }, step=step_global)      
wandb.log({"Test_eae_Identification_Head_F1":eae_report["Identification"]["Head"]["F1"] }, step=step_global)
wandb.log({"Test_eae_Identification_Coref_F1":eae_report["Identification"]["Coref"]["F1"] }, step=step_global)
wandb.log({"Test_eae_Classification_Head_F1":eae_report["Classification"]["Head"]["F1"] }, step=step_global)
wandb.log({"Test_eae_Classification_Coref_F1":eae_report["Classification"]["Coref"]["F1"] }, step=step_global)  
        

#%%
