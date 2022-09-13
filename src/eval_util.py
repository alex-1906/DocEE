from src.util import safe_div, compute_f1
import pandas as pd

def get_df(event_list,token_maps,doc_id_list):
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

    
    df = df.loc[doc_id_list]
    return df
            
def get_eval(event_list,token_maps,doc_id_list):

    df = get_df(event_list,token_maps,doc_id_list)

    idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
    clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0

    trigger_pred, trigger_gold, trigger_idf, trigger_clf = 0,0,0,0

    span_matches = []
    coref_span_matches = []
    trigger_matches = []
    counter = 0
    for idx, row in df.iterrows(): 

        events = row['pred_events']      
        gold_events = row['event_mentions']

        if counter < 3:
            if len(events) > 0:
                for ge,e in zip(gold_events,events):
                    print("----------- Gefundenes Event --------")
                    print(e)
                    print()
                    print("----------- Gold Event --------")
                    print(ge)
                    counter += 1
                    print("-----------------------------------")

        for e in events:
            for ge in gold_events:
                # ----- Trigger ------
                if ge['trigger']['start'] == e['trigger']['start'] and ge['trigger']['end'] == e['trigger']['end']:
                    trigger_idf += 1

                    print(f"!!!Trigger Match!!! {e['trigger']['text']}")
                    
                    if ge['event_type'] == e['event_type']:
                        trigger_clf += 1
                    # ----- Arguments ------
                    for g_arg in ge['arguments']:
                        for arg in e['arguments']:
                            #----- Head Matches -----
                            if g_arg['start'] == arg['start'] and g_arg['end'] == arg['end']:
                                span_matches.append((arg['entity_id'],(arg['start'],arg['end']),arg['text'],(g_arg['start'],g_arg['end']),g_arg['text']))
                                idf_h_matched += 1
                                idf_c_matched += 1 #head match bedeutet auch coref match, heads wurden extra aus corefs rausgenommen, damit nicht doppelt gezählt wird

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

        #Anzahl an predictions
        for e in events:
    	    for arg in e['arguments']:
                idf_pred += 1
                clf_pred += 1
        trigger_pred += len(events)
        #Anzahl an golds
        for ge in gold_events:
    	    for g_arg in ge['arguments']:
                idf_gold += 1
                clf_gold += 1
        trigger_gold += len(gold_events)
            
    print(f"\n# predicted triggers: {trigger_pred}")
    print(f"# gold triggers: {trigger_gold}")
    print(f"# matched triggers: {trigger_idf}")

    #----- Identification P,R,F1 -----
    idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
    idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

    #----- Classification P,R,F1 -----
    clf_h_p, clf_h_r, clf_h_f1 = compute_f1(idf_pred, idf_gold, clf_h_matched)
    clf_c_p, clf_c_r, clf_c_f1 = compute_f1(idf_pred, idf_gold, clf_c_matched)

    #----- Trigger Idf,Clf P,R,F1 -----
    trigger_idf_p, trigger_idf_r, trigger_idf_f1 = compute_f1(trigger_pred, trigger_gold, trigger_idf)
    trigger_clf_p, trigger_clf_r, trigger_clf_f1 = compute_f1(trigger_pred, trigger_gold, trigger_clf)


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
        },
        'Trigger':{
            'Identification':{
                'Matches':round(trigger_idf,2),
                'Precision':round(trigger_idf_p,2),
                'Recall':round(trigger_idf_r,2),
                'F1':round(trigger_idf_f1,2)
            },
            'Classification':{
                'Matches':round(trigger_clf,2),
                'Precision':round(trigger_clf_p,2),
                'Recall':round(trigger_clf_r,2),
                'F1':round(trigger_clf_f1,2)
            }
        }
    }
    return report
# %%
