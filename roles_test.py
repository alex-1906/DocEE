#%%
import pandas as pd
import json
from src.data import collate_fn, parse_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# %%
unique_roles = pd.read_json("data/Ontology/roles_unique.json")
shared_roles = pd.read_json("data/Ontology/roles_shared.json")

# %%
shared_roles = True
# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

if shared_roles:
    with open("data/Ontology/roles_shared.json") as f:
        relation_types = json.load(f)
else:
    with open("data/Ontology/roles_shared.json") as f:
        relation_types = json.load(f)
train_loader = DataLoader(
    parse_file("data/WikiEvents/preprocessed/train.json"),
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=9,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn)
# %%
sample = next(iter(train_loader))