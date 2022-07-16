import pandas as pd
import json
from nltk.tokenize import sent_tokenize, word_tokenize
#%%
train_path = "train.json"
test_path = "test.json"
dev_path = "dev.json"
#%%
def getVertex(row):
    vertexSet = []
    for em in row.event_mentions:
        t = em["trigger"]
        et = em['event_type']
        vertexSet.append([{
            'pos':[t["start"],t["end"]],
            'type':et+"."+"TRIGGER",
            'sent_id':t["sent_idx"],
            'name':t["text"],
            'id':em["id"]
        }])
        for a in em["arguments"]:
            #find argument_entity by given id
            arg_id = a['entity_id']

            for ent in row.entity_mentions:
                if(ent['id'] == arg_id):
                    vertexSet.append([{
                        'pos':[ent["start"],ent["end"]],
                        'type':ent["entity_type"],
                        'sent_id':ent["sent_idx"],
                        'name':ent["text"],
                        'id':ent["id"]
                    }])       
    return vertexSet
def getVertexAllEntities(row):
    vertexSet = []
    for em in row.event_mentions:
        t = em["trigger"]
        et = em['event_type']
        vertexSet.append([{
            'pos':[t["start"],t["end"]],
            'type':et+"."+"TRIGGER",
            'sent_id':t["sent_idx"],
            'name':t["text"],
            'id':em["id"]
        }])
    for ent in row.entity_mentions:
        vertexSet.append([{
            'pos':[ent["start"],ent["end"]],
            'type':ent["entity_type"],
            'sent_id':ent["sent_idx"],
            'name':ent["text"],
            'id':ent["id"]
        }])       
    return vertexSet
def getLabels(row):
    labels = []
    for event in row.event_mentions:
        for argument in event['arguments']:
            head = -1
            tail = -1
            for idx, entity in enumerate(row.vertexSet):
                #Achtung, sobald es mehrere mentions zu einer entity gibt klappt [0] nicht mehr!!!
                if(entity[0]['id'] == event['id']):
                    head = idx
                if(entity[0]['id'] == argument['entity_id']):
                    tail = idx
            labels.append({
                #r': event['event_type']+"."+argument['role'],
                'r': argument['role'],
                'h': head,
                't': tail,
                'evidence':[]
            })
    return labels
def getSents(row):
    sents = []
    for sent in row.sentences:
        sent_tokens = []
        for t in sent[0]:
            sent_tokens.append(t[0])
        sents.append(sent_tokens)
    return sents

#%%
for path in [train_path, test_path, dev_path]:
    with open(f"data/WikiEvents/{path}l") as f:
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']
        df_inter['json_element'].apply(json.loads)
        df_we = pd.json_normalize(df_inter['json_element'].apply(json.loads))

    df_we["vertexSet"] = df_we.apply(getVertexAllEntities,axis=1)
    df_we["labels"] = df_we.apply(getLabels,axis=1)
    df_we["title"] = df_we["doc_id"]
    df_we["sents"] = df_we.apply(getSents,axis=1)

    df_we.drop(columns=['text', 'sentences','relation_mentions', 'event_mentions'],inplace=True)
    df_we.to_json(f"data/WikiEvents/DocRed_Format/AllEntities/{path}",orient="records")

#%%
