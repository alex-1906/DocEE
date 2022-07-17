
#end span exklusive
def generate_spans_naive(text, max_len=21):
    spans = []
    for sent in text:
        length = len(sent)
        spans_sent = []
        #Wird so das letzte token überhaupt als einzelner span in Betracht gezogen?
        for i in range(0, length):
            for j in range(i, min(i+max_len, length)):
                spans_sent.append([i, j+1])
        spans.append(spans_sent)
    return spans

def generate_candidate_spans(text,max_len=21):
    spans = []
    for sent in text:
        length = len(sent)
        spans_sent = []
        for i in range(0, length):
            for j in range(i, min(i + max_len, length)):
                spans_sent.append([i, j+1])
        spans.append(spans_sent)
    return spans
