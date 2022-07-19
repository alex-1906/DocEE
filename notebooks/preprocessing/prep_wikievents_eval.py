import pandas as pd
import json
from collections import defaultdict



#%%
# ------ Load Data ------
input_file = "data/WikiEvents/train.jsonl"
coref_file = "data/WikiEvents/train_coref.jsonl"

with open(input_file) as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads)) 
    df_we = df_we.set_index('doc_id')
with open(coref_file) as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_co = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    df_co = df_co.set_index('doc_key')

#%% 
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
                        continue
                    if c == ent['id']:
                        coref = {
                            'entity_id':ent['id'],
                            'start':ent['start'],
                            'end':ent['end'],
                            'text': ent['text']
                        }
                        corefs.append(coref)
            arg['corefs'] = corefs
# %%
# ------ Save DF ------
df_we.to_json('data/WikiEvents/train_eval.json')
# %%
df_we
# %%
df = pd.read_json('data/WikiEvents/train_eval.json')
df
# %%
