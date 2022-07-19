from src.util import safe_div, compute_f1
import pandas as pd

def get_df(event_list,token_maps,doc_id_list):
    #df = pd.read_json('data/Dump/train_eval.json')
    df = pd.read_json('data/WikiEvents/preprocessed/full_eval.json')

    df['pred_events'] = pd.Series(event_list,index=doc_id_list)
    df['t_map'] = pd.Series(token_maps,index=doc_id_list)

    df['pred_events'] = df['pred_events'].fillna("").apply(list)
    df['t_map'] = df['t_map'].fillna("").apply(list)

    # ----- Adjust Indexing for Gold Events -----
    l_offset = 1
    for idx, row in df.iterrows():
        if len(row['t_map']) == 0:
            continue
        t_map = row.t_map
        for em in row.event_mentions:
            em['trigger']['start'] = t_map[em['trigger']['start']+1]
            em['trigger']['end'] = t_map[em['trigger']['end']+1]
            for arg in em['arguments']:
                arg['start'] = t_map[arg['start']+1]
                arg['end'] = t_map[arg['end']+1]
                for coref in arg['corefs']:
                    coref['start'] = t_map[coref['start']+1]
                    coref['end'] = t_map[coref['end']+1]
    return df


def get_eval(event_list,token_maps,doc_id_list):

    df = get_df(event_list,token_maps,doc_id_list)

    idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
    clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0
    span_matches = []
    coref_span_matches = []
    for idx, row in df.iterrows(): 
        events = row['pred_events']
        gold_events = row['event_mentions']

        for ge,e in zip(gold_events,events):
            for g_arg in ge['arguments']:
                for arg in e['arguments']:
                    #----- Head Matches -----
                    if g_arg['start'] == arg['start'] and g_arg['end'] == arg['end']:
                        span_matches.append((arg['entity_id'],(arg['start'],arg['end']),arg['text'],(g_arg['start'],g_arg['end']),g_arg['text']))
                        idf_h_matched += 1
                        idf_c_matched += 1
                        if g_arg['role'] == arg['role']:
                            clf_h_matched += 1
                            clf_c_matched += 1
                        
                    #----- Coref Matches -----
                    for coref in g_arg['corefs']:
                        if coref['start'] == arg['start'] and coref['end'] == arg['end']:
                            coref_span_matches.append((arg['entity_id'],(arg['start'],arg['end']),arg['text'],(coref['start'],coref['end']),coref['text']))
                            idf_c_matched += 1
                            if g_arg['role'] == arg['role']:
                                clf_c_matched += 1
                idf_gold += 1
            for arg in e['arguments']:
                idf_pred += 1
            clf_pred, clf_gold = idf_pred, idf_gold
    
    #----- Identification P,R,F1 -----
    idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
    idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

    #----- Classification P,R,F1 -----
    clf_h_p, clf_h_r, clf_h_f1 = compute_f1(idf_pred, idf_gold, clf_h_matched)
    clf_c_p, clf_c_r, clf_c_f1 = compute_f1(idf_pred, idf_gold, clf_c_matched)


    report = {
        'Identification':{
            'Head':{
                'Matches':round(idf_h_matched,2),
                'Precision':round(idf_h_p,2),
                'Recall':round(idf_h_r,2),
                'F1':round(idf_h_f1,2)
            },
            'Coref':{
                'Matches':round(idf_c_matched,2),
                'Precision':round(idf_c_p,2),
                'Recall':round(idf_c_r,2),
                'F1':round(idf_c_f1,2)
            }
        },
        'Classification':{
            'Head':{
                'Matches':round(clf_h_matched,2),
                'Precision':round(clf_h_p,2),
                'Recall':round(clf_h_r,2),
                'F1':round(clf_h_f1,2)
            },
            'Coref':{
                'Matches':round(clf_c_matched,2),
                'Precision':round(clf_c_p,2),
                'Recall':round(clf_c_r,2),
                'F1':round(clf_c_f1,2)
            }
        }
    }
    return report
# %%
# ----- Evaluation by Id -----
def get_eval_by_id(event_list,token_maps,doc_id_list):

    df = get_df(event_list,token_maps,doc_id_list)

    idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
    clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0
    span_matches = []
    coref_span_matches = []
    for idx, row in df.iterrows(): 
        events = row['pred_events']
        gold_events = row['event_mentions']

        for ge,e in zip(gold_events,events):
            for g_arg in ge['arguments']:
                for arg in e['arguments']:
                    #----- Head Matches -----
                    if g_arg['entity_id'] == arg['entity_id']:
                        span_matches.append((arg['entity_id'],(arg['start'],arg['end']),arg['text'],(g_arg['start'],g_arg['end']),g_arg['text']))
                        idf_h_matched += 1
                        idf_c_matched += 1
                        if g_arg['role'] == arg['role']:
                            clf_h_matched += 1
                            clf_c_matched += 1
                        
                    #----- Coref Matches -----
                    for coref in g_arg['corefs']:
                        if coref['entity_id'] == arg['entity_id']:
                            coref_span_matches.append((arg['entity_id'],(arg['start'],arg['end']),arg['text'],(coref['start'],coref['end']),coref['text']))
                            idf_c_matched += 1
                            if g_arg['role'] == arg['role']:
                                clf_c_matched += 1
                idf_gold += 1
            for arg in e['arguments']:
                idf_pred += 1
            clf_pred, clf_gold = idf_pred, idf_gold
    
    #----- Identification P,R,F1 -----
    idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
    idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

    #----- Classification P,R,F1 -----
    clf_h_p, clf_h_r, clf_h_f1 = compute_f1(idf_pred, idf_gold, clf_h_matched)
    clf_c_p, clf_c_r, clf_c_f1 = compute_f1(idf_pred, idf_gold, clf_c_matched)


    report = {
        'Identification':{
            'Head':{
                'Matches':round(idf_h_matched,2),
                'Precision':round(idf_h_p,2),
                'Recall':round(idf_h_r,2),
                'F1':round(idf_h_f1,2)
            },
            'Coref':{
                'Matches':round(idf_c_matched,2),
                'Precision':round(idf_c_p,2),
                'Recall':round(idf_c_r,2),
                'F1':round(idf_c_f1,2)
            }
        },
        'Classification':{
            'Head':{
                'Matches':round(clf_h_matched,2),
                'Precision':round(clf_h_p,2),
                'Recall':round(clf_h_r,2),
                'F1':round(clf_h_f1,2)
            },
            'Coref':{
                'Matches':round(clf_c_matched,2),
                'Precision':round(clf_c_p,2),
                'Recall':round(clf_c_r,2),
                'F1':round(clf_c_f1,2)
            }
        }
    }
    return report
# %%




def get_eae_eval(doc_ids_list,event_list):
    df_we = pd.read_json('../data/WikiEvents/train.json').set_index('doc_id')
    coref_mapping = pd.read_json('../data/WikiEvents/coref_mapping.json')

    idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
    clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0
    for doc_id, events in zip(doc_ids_list,event_list):
        gold_events = df_we.loc[doc_id].event_mentions

        
        for ge,e in zip(gold_events,events):
            for g_arg in ge['arguments']:
                for arg in e['arguments']:
                    #----- Head Matches -----
                    if g_arg['entity_id'] == arg['entity_id']:
                        idf_h_matched += 1
                        if g_arg['role'] == arg['role']:
                            clf_h_matched += 1
                        
                    #----- Coref Matches -----
                    if arg['entity_id'] in coref_mapping[doc_id][g_arg['entity_id']]:
                        idf_c_matched += 1
                        #wenn roles übereinstimmen
                        if g_arg['role'] == arg['role']:
                            clf_c_matched += 1
                                
                idf_gold += 1
            #zaehle alle erkannten
            for arg in e['arguments']:
                idf_pred += 1

            idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
            idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

            return idf_h_p, idf_h_r, idf_h_f1, idf_c_p, idf_c_r, idf_c_f1
    
            
                