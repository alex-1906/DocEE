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

#%%
wandb.init(project='test-remote-cpu')

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
#########################
## Model Configuration ##
#########################
torch.use_deterministic_algorithms(True)
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
    batch_size=2,
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
                soft_mention=False,
                at_inference=False,
                 )
optimizer = AdamW(mymodel.parameters(), lr=1e-5, eps=1e-6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
torch.save(mymodel.state_dict(), "checkpoints/mymodel.pt")
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
