#%%
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
import torch.nn.functional as F
from src.losses import ATLoss
from src.util import process_long_input
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
from model import Encoder

#%%
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


with open("data/roles.json") as f:
    relation_types = json.load(f)

max_n = 9
test_loader = DataLoader(

    parse_file("data/WikiEvents/train_docred_format_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn)

mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                )
mymodel.eval()
#%%

doc_ids_list = []
event_list = []
eae_event_list = []
with tqdm.tqdm(test_loader) as progress_bar:
    for sample in progress_bar:

        input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample

        if input_ids.size(1) > 4096:
            # skip an insanely long document...
            continue

        #pass data to model for end-to-end task 
        _ ,_ , events = mymodel(input_ids, attention_mask, candidate_spans)
        #TODO: Performance Auswertung für schweren Task


        #pass data to model for eae task 
        _ ,_ , eae_events = mymodel(input_ids, attention_mask, candidate_spans, relation_labels = relation_labels, entity_spans = entity_spans, entity_types = entity_types, entity_ids = entity_ids)
        #TODO: Performance Auswertung für leichten Task

        for e in events:
            event_list.append(e)
        for ea in eae_events:
            eae_event_list.append(ea) 
        for d in doc_ids:
            doc_ids_list.append(d)

# %% 
