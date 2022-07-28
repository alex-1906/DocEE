#%%
import pandas as pd
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from src.docred_util import to_docred
#%%
#------- Load Raw WikiEvent Files --------#
def preprocess_we(coref = False):
    if coref:
        train_path = "coref/train.json"
        test_path = "coref/test.json"
        dev_path = "coref/dev.json"
    else:
        train_path = "train.json"
        test_path = "test.json"
        dev_path = "dev.json"

    df_list = []
    df_list_raw = []
    for path in [train_path, test_path, dev_path]:
        with open(f"data/WikiEvents/raw/{path}l") as f:
            lines = f.read().splitlines()
            df_inter = pd.DataFrame(lines)
            df_inter.columns = ['json_element']
            df_inter['json_element'].apply(json.loads)
            df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
            df_list_raw.append(df.copy(deep=True))
            df_list.append(df)

            if not coref:

                if path == train_path:
                    df['event_len'] = df['event_mentions'].apply(lambda x: len(x))
                    df.sort_values(by='event_len',inplace=True)
                    df_small = df[:18]
                    df_medium = df[12:60]

                    df_small = to_docred(df_small)
                    df_medium = to_docred(df_medium)

                    df_small.to_json(f"data/WikiEvents/preprocessed/train_small.json",orient="records")
                    df_medium.to_json(f"data/WikiEvents/preprocessed/train_medium.json",orient="records")

                df = to_docred(df)
                df.to_json(f"data/WikiEvents/preprocessed/{path}",orient="records")
            else:
                df.to_json(f"data/WikiEvents/preprocessed/{path}",orient="records")
    df_we = pd.concat(df_list)
    df_we_raw = pd.concat(df_list_raw)
    if not coref:
        df_we.to_json(f"data/WikiEvents/preprocessed/full.json",orient="records")
        df_we_raw.to_json(f"data/WikiEvents/raw/full.json",orient="records")
    else:
        df_we.to_json(f"data/WikiEvents/preprocessed/coref/full.json",orient="records")
        df_we_raw.to_json(f"data/WikiEvents/raw/coref/full.json",orient="records")
    
#%%
#------- Prep WikiEvents for Eval by inserting Corefs--------#
def prep_we_eval():
    df_we = pd.read_json("data/WikiEvents/raw/full.json").set_index('doc_id')
    df_co = pd.read_json("data/WikiEvents/raw/coref/full.json").set_index('doc_key')

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
    df_we.to_json('data/WikiEvents/preprocessed/full_eval.json')


#%%
#------- Roles --------#
def get_roles_file(shared=True):
    df = pd.read_json("data/WikiEvents/preprocessed/full.json")

    def get_roles(row):
        roles = []
        for e in row.labels:
            if shared:
                roles.append(e['r'].split('.')[-1])
            else:
                roles.append(e['r'])
        return roles
    df["roles"] = df.apply(get_roles,axis=1)
    #%%
    roles = []
    for idx,row in df.iterrows():
        for r in row.roles:
            roles.append(r)
    roles_set = list(set(roles))
    #roles_set.insert(0,"NOTA")
    if shared:
        path = "data/Ontology/roles_shared.json"
    else:
        path = "data/Ontology/roles_unique.json"
    with open(path, "w") as f:
        json.dump(roles_set,f)
#%%

#%%
def get_feasible_roles_file():
    df = pd.read_json("data/WikiEvents/raw/full.json")
    event_types = []
    for _,row in df.iterrows():
        for em in row.event_mentions:
            event_types.append(em['event_type'])
    event_types = list(set(event_types))
    #%%
    #--------- Feasible Roles ----------

    kairos = pd.read_json("data/Ontology/event_role_KAIROS.json")
    kairos = kairos.transpose()
    problems = []
    for i in event_types:
        if i not in kairos.index:
            problems.append(i)
    #print(problems)

    #manually lookup roles for new created event types in WikiEvents
    #insert most fitting KAIROS roles 
    #https://github.com/raspberryice/gen-arg/blob/main/aida_ontology_cleaned.csv

    row1 = kairos.loc['Contact.RequestCommand.Unspecified']
    row2 = kairos.loc['Contact.RequestCommand.Unspecified']
    row3 = kairos.loc['Contact.ThreatenCoerce.Unspecified']
    row4 = kairos.loc['Contact.RequestCommand.Unspecified']
    row5 = kairos.loc['Contact.ThreatenCoerce.Unspecified']
    row1.name = 'Contact.RequestCommand.Broadcast'
    row2.name = 'Contact.RequestCommand.Meet'
    row3.name = 'Contact.ThreatenCoerce.Broadcast'
    row4.name = 'Contact.RequestCommand.Correspondence'
    row5.name = 'Contact.ThreatenCoerce.Correspondence'
    kairos = kairos.append([row1,row2,row3,row4,row5])

    feasible_roles = defaultdict(dict)
    for idx,row in kairos.iterrows():
        feasible_roles[idx] = row.roles
    with open("data/Ontology/feasible_roles.json", "w") as f:
        json.dump(feasible_roles,f)
#%%
#--------- Mention Types ----------
def get_mention_types_file():
    df = pd.read_json("data/WikiEvents/raw/full.json")

    all_entities = []
    event_types = []
    for _,row in df.iterrows():
        for mention in row.entity_mentions:
            all_entities.append(mention['entity_type'])
        for em in row.event_mentions:
            event_types.append(em['event_type']+".TRIGGER")

    entity_types = list(set(all_entities))
    event_types = list(set(event_types))
    mention_types = entity_types + event_types
    
    with open("data/Ontology/mention_types.json", "w") as f:
        json.dump(mention_types,f)

#%%

