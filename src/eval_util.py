from src.util import safe_div,compute_f1
import pandas as pd

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
    
            
                