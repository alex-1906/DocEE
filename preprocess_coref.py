#%%
import pandas as pd
import json
from collections import defaultdict
from src.docred_util import to_docred




#%%


with open(f"data/WikiEvents/raw/train.jsonl") as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads)).set_index('doc_id')
with open("data/WikiEvents/raw/coref/train.jsonl") as f:
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

df_we.drop(index='backpack_ied_0',inplace=True)
df_we['event_len'] = df_we['event_mentions'].apply(lambda x: len(x))
df_we.sort_values(by='event_len',inplace=True)

df_small = df_we[:18]
df_medium = df_we[12:60]
df_large = df_we.copy(deep=True)

df_small = to_docred(df_small,coref=True)
df_medium = to_docred(df_medium,coref=True)
df_large = to_docred(df_large,coref=True)
df_we = to_docred(df_we,coref=True)
#drop docs without relations
df_large['len'] = df_large['labels'].apply(lambda x: len(x))
df_large = df_large.drop(df_large[df_large['len']==0].index)


df_small.to_json(f"data/WikiEvents/preprocessed/coref/train_small.json",orient="records")
df_medium.to_json(f"data/WikiEvents/preprocessed/coref/train_medium.json",orient="records")
df_large.to_json(f"data/WikiEvents/preprocessed/coref/train_large.json",orient="records")
df_we.to_json(f"data/WikiEvents/preprocessed/coref/train.json",orient="records")


#%%

with open(f"data/WikiEvents/raw/dev.jsonl") as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads)).set_index('doc_id')
with open(f"data/WikiEvents/raw/coref/dev.jsonl") as f:
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

df_we = to_docred(df_we,coref=True)

df_we.to_json(f"data/WikiEvents/preprocessed/coref/dev.json",orient="records")

#%%
with open(f"data/WikiEvents/raw/test.jsonl") as f:
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads)).set_index('doc_id')
with open(f"data/WikiEvents/raw/coref/test.jsonl") as f:
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


df_we = to_docred(df_we,coref=True)

df_we.to_json(f"data/WikiEvents/preprocessed/coref/test.json",orient="records")