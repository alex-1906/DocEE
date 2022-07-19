#%%
import argparse
import pandas as pd
import json
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          DistilBertConfig, RobertaConfig, XLMRobertaConfig)
from transformers.optimization import AdamW
from src.data import collate_fn, parse_file
from src.eval_util import get_eval, get_eval_by_id
from src.full_model import Encoder
from src.util import set_seed
import wandb
import argparse
import random
import string

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2, help="eval batch size")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
parser.add_argument("--model", type=str, default="base", help="model")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--project", type=str, default="test", help="project name for wandb")
parser.add_argument("--soft_mention", type=str, default=False, help="method for mention detection")
parser.add_argument("--at_inference", type=str, default=False, help="use at labels for inference")

#%%
args = parser.parse_args()
wandb.init(project=args.project)
wandb.config.update(args)
wandb.config.identifier = random_string

wandb.config = {
  "learning_rate": 1e-5,
  "epochs": 1,
  "batch_size": 2,
  "shuffle": False
}
torch.use_deterministic_algorithms(True)
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


with open("data/Ontology/roles.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/trigger_entity_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
train_loader = DataLoader(
    #parse_file("data/WikiEvents/DocRed_Format/AllEntities/train.json",
    parse_file("data/WikiEvents/DocRed_Format/train_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
    shuffle=False,
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
optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate, eps=1e-6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

mymodel.to(device)

# %%
# ---------- Train Loop -----------#
mymodel.train()

losses = []
event_list = []
doc_id_list = []
token_maps = []

with tqdm.tqdm(train_loader) as progress_bar:
    for sample in progress_bar:
        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        
        # --------- E2E Task  ------------#
        mention_loss,argex_loss,loss,e2e_events = mymodel(input_ids.to(device), attention_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        progress_bar.set_postfix({"L":f"{loss.item():.2f}"})
        wandb.log({"e2e_train_loss": loss.item()})
        wandb.log({"e2e_mention_loss": mention_loss.item()})
        wandb.log({"e2e_argex_loss": argex_loss.item()})

        for batch_i in range(input_ids.shape[0]):
            doc_id_list.append(doc_ids[batch_i])
            token_maps.append(token_map[batch_i])
            event_list.append(e2e_events[batch_i])

# %%
e2e_report = get_eval(event_list,token_maps,doc_id_list)
torch.save(mymodel.state_dict(), f"checkpoints/{args.checkpoint}.pt")
table = wandb.Table(columns=["Idf_head_F1","Idf_coref_F1","Clf_head_F1","Clf_coref_F1","Head_Matches","Coref_Matches"])
table.add_data(e2e_report["Identification"]["Head"]["F1"],
            e2e_report["Identification"]["Coref"]["F1"],
            e2e_report["Classification"]["Head"]["F1"],
            e2e_report["Classification"]["Coref"]["F1"],
            e2e_report["Identification"]["Head"]["Matches"],
            e2e_report["Identification"]["Coref"]["Matches"])
wandb.log({"mytable": table})

print(e2e_report)
#%%
