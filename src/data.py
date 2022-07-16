import json
from tqdm import tqdm
import transformers
import numpy as np
import torch
from src.candidate_generation.naive_candidates import generate_spans_naive, generate_candidate_spans
from src.candidate_generation.util import sent_ids_to_token_ids
from transformers import AutoTokenizer


def tokenization_and_vectorization(tokens, entities, tokenizer):
    """
    --- Input ---
    tokens: list of strings (word tokens)
    entities: list containing list of tuples per entity. each touple marks a mention span.
    tokenizer: tokenizer for final tokenization of word tokens into subtokens
    """

    # flatten mention span list
    mention_spans = [(mention, i) for i, ent in enumerate(entities) for mention in ent]

    # for each token create a binary vector indicating to which mention(s) it belongs
    tmap = np.zeros((len(tokens), len(mention_spans)))
    for i, span in enumerate(mention_spans):
        start, end = span[0]
        tmap[start:end, i] = 1
    tmap = tmap.tolist()

    token_strings = []
    start_tokens = [None for _ in mention_spans]
    end_tokens = [None for _ in mention_spans]

    # iterate over word tokens and build output
    state = np.zeros(len(mention_spans)).tolist()
    # append blank tokens to end so we can close spans including the last token
    tmap.append(np.zeros(len(mention_spans)).tolist())
    tokens.append("")
    # token_map will contain the index the start token for each word token to allow for mapping from word index to token index (needed for candidate span generation)
    token_map = []
    for token, map in zip(tokens, tmap):
        token_map.append(len(token_strings))
        # check for state changes
        spans_started = []
        spans_ended = []
        if map != state:
            for i, (prev, next) in enumerate(zip(state, map)):
                if prev == next:
                    pass
                elif prev > next:
                    # span has ended
                    spans_ended.append(i)                
                elif prev < next:
                    # span has ended
                    spans_started.append(i)

        # insert end tokens
        for span_id in spans_ended:
            end_tokens[span_id] = len(token_strings)
        
        # insert start tokens
        for span_id in spans_started:
            start_tokens[span_id] = len(token_strings)
        
        # create subtokens
        for sub_token in tokenizer.tokenize(token):
            token_strings.append(sub_token)
                        
        # carry over state
        state = map

    token_ids = tokenizer.convert_tokens_to_ids(token_strings)

    if None in end_tokens or None in start_tokens:
        print("Problem!")

    mentions = [(s,e) for s,e in zip(start_tokens, end_tokens)]

    entities = [[] for _ in entities]

    for ms, mention in zip(mention_spans, mentions):
        entities[ms[1]].append(mention)

    return token_ids, entities, token_map


def parse_file(filepath, tokenizer, relation_types, max_candidate_length=3):
    # open file
    input_file = json.load(open(filepath, "r"))
    output = []

    description = f"Parsing & generating candidates (n={max_candidate_length})"

    for sample in tqdm(input_file, desc=description):
        # parse a single wikievents sample
        tokens = sample['tokens']
        sentences = sample['sents']
        # get labeled entity mention spans
        vertices = sample['vertexSet']
        doc_id = sample['doc_id']

        if type(tokenizer) == transformers.BertTokenizer or type(tokenizer) == transformers.BertTokenizerFast:
            l_offset = 1
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        else:
            l_offset = 0

        entities = []
        entity_types = []
        entity_ids = []
        for vertex in vertices:
            if type(vertex) is not list:
                vertex = [vertex]
            mentions = []
            for mention in vertex:
                start, end = mention['pos']
                #Bei WikiEvents wird nicht bei jedem Satz neu angefangen zu zaehlen.
                start += l_offset
                end += l_offset
                mentions.append((start, end))
                entity_types.append(mention['type'])
                entity_ids.append(mention['id'])
            entities.append(mentions)

        token_ids, entity_spans, token_map = tokenization_and_vectorization(tokens, entities, tokenizer)

        relation_labels = {}

        for rel in sample['labels']:
            pair = (rel['h'], rel['t'])
            label = relation_types.index(rel['r'])
            relation_labels[pair] = label
            #triple = (rel['h'], rel['t'], label)
            #relation_labels.append(triple)
            '''if pair not in relation_labels.keys():
                relation_labels[pair] = [0] * len(relation_types)
            
            relation_labels[pair][label] = 1'''


        # generate candidate spans
        candidate_spans = generate_spans_naive(sentences,max_len=max_candidate_length)
        # convert indexing to token level
        candidate_spans = sent_ids_to_token_ids(sentences, token_map, candidate_spans)

        #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        #text = tokenizer.convert_ids_to_tokens(token_ids)

        output.append({
            'title': sample['title'],
            'text': sentences,
            'token_map': token_map,
            'token_ids': token_ids,
            'entity_spans': entity_spans,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'candidate_spans': candidate_spans,
            'relation_labels': relation_labels,
            'doc_id':doc_id
        })


    return output


def collate_fn(batch):

    # get max dimensions for padding
    max_len = max([len(f["token_ids"]) for f in batch])

    # pad token_ids and create a mask to indicate padding
    token_ids = [f["token_ids"] + [0] * (max_len - len(f["token_ids"])) for f in batch]
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f["token_ids"]) + [0.0] * (max_len - len(f["token_ids"])) for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    # merge other info in batch
    token_map = [f["token_map"] for f in batch]
    text = [f["text"] for f in batch]
    entity_spans = [f["entity_spans"] for f in batch]
    entity_types = [f["entity_types"] for f in batch]
    entity_ids = [f["entity_ids"] for f in batch]
    relation_labels = [f["relation_labels"] for f in batch]
    candidate_spans = [f["candidate_spans"] for f in batch]
    doc_id = [f["doc_id"] for f in batch]

    return token_ids, input_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_id

