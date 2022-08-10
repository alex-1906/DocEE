#%%
import pandas as pd
import json
from collections import defaultdict
from src.docred_util import to_docred

import tqdm

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          DistilBertConfig, RobertaConfig, XLMRobertaConfig)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.data import collate_fn, parse_file
from src.model import Encoder
#%%


with open(f"data/WikiEvents/raw/train.jsonl") as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads)).set_index('doc_id')
with open(f"data/WikiEvents/raw/coref/train.jsonl") as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_co = pd.json_normalize(df_inter['json_element'].apply(json.loads)).set_index('doc_key')


df_we['doc_id'] = df_we.index
# ------ Coref Mapping ------
coref_mapping = defaultdict(dict)
for doc_id, row in df_co.iterrows():
    for cluster in row.clusters:
        for item in cluster:
            coref_mapping[doc_id][item] = cluster
for doc_id, row in df_we.iterrows():
    for event in row.event_mentions:
        for arg in event['arguments']:
            if arg['entity_id'] not in coref_mapping[doc_id].keys():
                coref_mapping[doc_id][arg['entity_id']] = arg['entity_id']

    #%%
# ------ Insert Coreferences to DF ------
for idx,row in df_we.iterrows():
    for em in row.event_mentions:
        for arg in em['arguments']:
            coref_ids = coref_mapping[idx][arg['entity_id']]
            corefs = []
            for c in coref_ids:
                for ent in row.entity_mentions:
                    if arg['entity_id'] == ent['id']:
                        arg['start'] = ent['start']
                        arg['end'] = ent['end']
                        arg['entity_type'] = ent['entity_type']
                        continue
                    if c == ent['id']:
                        coref = {
                            'entity_id':ent['id'],
                            'start':ent['start'],
                            'end':ent['end'],
                            'text': ent['text'],
                            'entity_type':ent['entity_type'],
                            'sent_id':ent["sent_idx"]
                        }
                        corefs.append(coref)
            arg['corefs'] = corefs    
#%%
df_we.event_mentions[15]

for idx,row in df_we.iterrows():
    for e in row.event_mentions:
        for arg in e['arguments']:
            if len(arg['corefs'])>0 and len(e) < 4:
                print(idx)

#%%
#df_we.to_json('data/WikiEvents/preprocessed/full_eval.json')
df_we.loc['scenario_en_kairos_65'].event_mentions
# %%
df_we = to_docred(df_we,coref=True)

#%%
df_we.iloc[1].vertexSet
# %%
for idx,row in df_we.iterrows():
    for e in row.vertexSet:
        if len(e) > 3:
            print(e)
            print()
# %%

for e in df_we.loc['scenario_en_kairos_65'].vertexSet:
    if len(e) > 3:
        print(e)
        print()

len(df_we.loc['scenario_en_kairos_65'].vertexSet)
#%%
#Test ob es richtig durch den Train Loader kommt
df_we.to_json(f"data/WikiEvents/preprocessed/coref_test.json",orient="records")

with open("data/Ontology/roles_shared.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/mention_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

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

train_loader = DataLoader(
    parse_file("data/WikiEvents/preprocessed/coref_test.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=9),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn)
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
#%%

for sample in train_loader:
    input_ids, attention_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
    for d in doc_ids:
        if d == 'scenario_en_kairos_65':
            print(d)
            print(entity_spans[0][2])
            print(entity_spans[0][1])
            print(len(entity_spans[0][2]))

            _,_,_,eae_events = mymodel(input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, text, e2e=False)

            
# %%
# %%
