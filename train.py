#%%
import argparse
import pandas as pd
import json
import tqdm
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          DistilBertConfig, RobertaConfig, XLMRobertaConfig)
from transformers.optimization import AdamW
from src.data import collate_fn, parse_file
from src.eval_util import get_eval, get_eval_by_id
from src.full_model import Encoder
import wandb

#%%
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


with open("data/Ontology/roles.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/trigger_entity_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
train_loader = DataLoader(
    parse_file("data/WikiEvents/DocRed_Format/train_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn)
dev_loader = DataLoader(
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
        mention_loss,argex_loss,loss,e2e_events = mymodel(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        progress_bar.set_postfix({"L":f"{loss.item():.2f}"})
        wandb.log({"e2e_train_loss": loss.item()})
        wandb.log({"e2e_mention_loss": mention_loss.item()})
        wandb.log({"e2e_argex_loss": argex_loss.item()})

# %%
# ---------- Eval on Dev -----------#

newmodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=False,
                at_inference=False,
                 )
newmodel.eval()
from src import model

eae_model = model.Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=False,
                at_inference=False,
                 )
eae_model.eval()
#%%
torch.save(eae_model.state_dict(), "checkpoints/eae_model.pt")
newmodel.load_state_dict(torch.load("checkpoints/eae_model.pt"))

#%%

event_list_e2e,event_list_eae,event_list_eae_2 = [],[],[]
doc_id_list = []
token_maps = []

with tqdm.tqdm(dev_loader) as progress_bar:
    for sample in progress_bar:
        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        
        # --------- E2E Task  ------------#
        _,_,_,e2e_events = newmodel(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=True)

        # --------- EAE Task  ------------#
        _,_,_,eae_events = newmodel(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)
        _,_,eae_events_2 = eae_model(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text)

        e2e_report = get_eval(event_list_e2e,token_maps,doc_id_list)
        eae_report = get_eval(event_list_eae,token_maps,doc_id_list)
        wandb.log({"e2e_report": e2e_report})
        wandb.log({"eae_report": eae_report})

        for batch_i in range(input_ids.shape[0]):
            doc_id_list.append(doc_ids[batch_i])
            token_maps.append(token_map[batch_i])
            event_list_e2e.append(e2e_events[batch_i])
            event_list_eae.append(eae_events[batch_i])
            event_list_eae_2.append(eae_events_2[batch_i])


# %%
e2e_report = get_eval(event_list_e2e,token_maps,doc_id_list)
eae_report = get_eval(event_list_eae,token_maps,doc_id_list)
eae_report_id = get_eval_by_id(event_list_eae,token_maps,doc_id_list)
eae_report_2 = get_eval(event_list_eae_2,token_maps,doc_id_list)
#%%
print(eae_report == eae_report_id)
print(eae_report == eae_report_2)
# %%
table = wandb.Table(columns=["Idf_head_F1","Idf_coref_F1","Clf_head_F1","Clf_coref_F1"])
table.add_data(eae_report["Identification"]["Head"]["F1"],eae_report["Identification"]["Coref"]["F1"],eae_report["Classification"]["Head"]["F1"],eae_report["Classification"]["Coref"]["F1"])
wandb.log({"mytable": table})
#%%
table = wandb.Table(eae_report)
table.add_data(eae_report)
wandb.log({"mytable": table})

#%%
get_eval(eae_events,token_map,doc_ids)
#%%
get_eval_by_id(events,token_maps,doc_id_list)
#%%
e2e_events
#%%
for em in event_list_eae:
    for arg in em['arguments']:
        print(arg)
#%%
for doc in event_list_eae:
    for em in doc:
        for arg in em['arguments']:
            print(arg)
#%%
with tqdm.tqdm(train_loader) as progress_bar:
    for sample in progress_bar:
        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
