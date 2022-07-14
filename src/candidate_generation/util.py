
def sent_ids_to_token_ids(text, token_map, spans, l_offset=1):
    lens = [0,]
    sen_len = 0
    for sentence in text:
        for token in sentence:
            sen_len += 1
        lens.append(sen_len)
    
    adjusted_spans = []
    for i, sent in enumerate(spans):
        for span in sent:
            start, end = span
            start += lens[i] + l_offset
            end += lens[i] + l_offset
            adjusted_spans.append((token_map[start], token_map[end]))


    return adjusted_spans

def prec_rec_f1(true_positives, false_negatives, false_positives):
    recall = true_positives/(true_positives+false_negatives)
    precision = true_positives/(true_positives+false_positives + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return recall,precision,f1

def evaluate_candidate_generation(candidates, truth):

    true_positives = 0
    false_negatives = 0

    for span in truth:
        if span in candidates:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = 0
    for span in candidates:
        if span not in truth:
            false_positives += 1 

    return true_positives,false_negatives,false_positives
