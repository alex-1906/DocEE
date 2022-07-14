import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel, BertConfig

from torch.utils.data import DataLoader
from src.data import parse_file, collate_fn
import tqdm
import json
from transformers.optimization import AdamW
import numpy as np
import torch.nn.functional as F
from src.losses import ATLoss
from src.util import process_long_input
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
#%%
class Encoder(nn.Module):
    def __init__(self, config, model, cls_token_id, sep_token_id):
        super().__init__()
        
        self.config = config
        self.model = model

        self.entity_anchor = nn.Parameter(torch.zeros((66, 768)))
        torch.nn.init.uniform_(self.entity_anchor, a=-1.0, b=1.0)
        
        self.relation_embeddings = nn.Parameter(torch.zeros((57,3*768)))
        torch.nn.init.uniform_(self.relation_embeddings, a=-1.0, b=1.0)            
        self.nota_embeddings = nn.Parameter(torch.zeros((20,3*768)))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)


        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.at_loss = ATLoss()

        self.k_mentions = 50
                
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        
        with open("roles.json") as f:
            self.relation_types = json.load(f)
        with open("trigger_entity_types.json") as f:
            self.types = json.load(f)
        with open("feasible_roles.json") as f:
            self.feasible_roles = json.load(f)
        
    def encode(self, input_ids, attention_mask):
        config = self.config
        if type(config) == BertConfig or type(config) == DistilBertConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id]
        elif type(config) == RobertaConfig or type(config) == XLMRobertaConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id, self.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention


   
        
    def forward(self, input_ids, attention_mask, candidate_spans, relation_labels = [[]], entity_spans= [[]], entity_types= [[]], entity_ids= [[]]):
        sequence_output, attention = self.encode(input_ids, attention_mask)

        loss = torch.zeros((1)).to(sequence_output)
        mention_loss = torch.zeros((1)).to(sequence_output)
        counter = 0
        batch_triples = []
        batch_events = []
        batch_text = []
        
        for batch_i in range(sequence_output.size(0)):
            text_i = self.tokenizer.convert_ids_to_tokens(input_ids[batch_i])
            batch_text.append(text_i)

            # MENTION DETECTION

            # ---------- Candidate span embeddings ------------
            mention_candidates = []
            candidates_attentions = []
            for span in candidate_spans[batch_i]:
                mention_embedding = torch.mean(sequence_output[batch_i, span[0]:span[1]+1,:], 0)
                mention_attention = torch.mean(attention[batch_i, span[0]:span[1]+1,:], 0)
                mention_candidates.append(mention_embedding)
                candidates_attentions.append(mention_attention)
            embs = torch.stack(mention_candidates)
            atts = torch.stack(candidates_attentions)

            # ---------- Soft mention detection (scores) ------------
            span_scores = embs.unsqueeze(1) * self.entity_anchor.unsqueeze(0)
            span_scores = torch.sum(span_scores, dim=-1)
            span_scores_max, class_for_span = torch.max(span_scores, dim=-1)
            scores_for_max, max_spans = torch.topk(span_scores_max.view(-1), min(self.k_mentions, embs.size(0)), dim=0)
            class_for_max_span = class_for_span[max_spans]

            #types = ['TRIGGER','VAL', 'ABS', 'SID', 'MHI', 'BOD', 'VEH', 'GPE', 'MON', 'PER', 'TTL', 'LOC', 'WEA', 'INF', 'COM', 'FAC', 'CRM', 'ORG']
            types = self.types
            

            if self.training:
                # ---------- Mention Loss and adding true spans during training ------------

                spans_for_type = {}

                for span, rtype in zip(entity_spans[batch_i], entity_types[batch_i]):
                    if rtype not in spans_for_type.keys():
                        spans_for_type[rtype] = []
                    spans_for_type[rtype].append(span[0])

                anchors, positives, negatives = [], [], []

                for rtype, positive_examples in spans_for_type.items():

                    # add negative examples from entity spans
                    for pos in positive_examples:
                        for rtype2, negative_examples in spans_for_type.items():
                            if rtype2 == rtype:
                                continue
                            for neg in negative_examples:
                                anchors.append(self.entity_anchor[types.index(rtype),:])
                                positives.append(torch.mean(sequence_output[batch_i, pos[0]:pos[1]+1,:], 0))
                                negatives.append(torch.mean(sequence_output[batch_i, neg[0]:neg[1]+1,:], 0))

                    # add negative examples from candidate spans
                    for pos in positive_examples:
                        for neg in [x for x in candidate_spans[batch_i] if x not in entity_spans[batch_i]]:
                            anchors.append(self.entity_anchor[types.index(rtype),:])
                            positives.append(torch.mean(sequence_output[batch_i, pos[0]:pos[1]+1,:], 0))
                            negatives.append(torch.mean(sequence_output[batch_i, neg[0]:neg[1]+1,:], 0))


                mention_loss += self.triplet_loss(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))


            # ARGUMENT ROLE LABELING


            if self.training:
                # ---------- Pooling Entity Embeddings and Attentions ------------
                entity_embeddings = []
                entity_attentions = []
                for ent in entity_spans[batch_i]:
                    ent_embedding = torch.mean(sequence_output[batch_i, ent[0][0]:ent[0][1],:],0)
                    entity_embeddings.append(ent_embedding)
                    ent_attention = torch.mean(attention[batch_i,:,ent[0][0]:ent[0][1],:],1)
                    entity_attentions.append(ent_attention)
                if(len(entity_embeddings) == 0):
                    continue
                entity_embeddings = torch.stack(entity_embeddings)
                entity_attentions = torch.stack(entity_attentions)
            else:
                entity_embeddings = embs[max_spans]
                entity_attentions = atts[max_spans]
                
                

                '''
                Wie mache ich das, dass beim training die entity_types,entity_spans,relation_labels mit übergeben werden aber bei der inferenz nicht?
                
                entity_types = []
                for c in class_for_max_span:
                    entity_types.append(types[c])
                    
                entity_spans = []
                for s in max_spans:
                    entity_spans.append(candidate_spans[batch_i][s])
                
                    
                    
                '''
                
            # ---------- Localized Context Pooling ------------
            relation_candidates = []
            localized_context = []
            concat_embs = []
            triggers = []
            for s in range(entity_embeddings.shape[0]):
                if entity_types[batch_i][s].split(".")[-1] != "TRIGGER":
                    continue
                triggers.append(s)
                for o in range(entity_embeddings.shape[0]):
                    if s != o:

                        relation_candidates.append((s,o))

                        A_s = entity_attentions[s,:,:]
                        A_o = entity_attentions[o,:,:]
                        A = torch.mul(A_o,A_s)
                        q = torch.sum(A,0)
                        a = q / q.sum()
                        H_T = sequence_output[batch_i].T
                        c = torch.matmul(H_T,a)
                        localized_context.append(c)

                        concat_emb = torch.cat((entity_embeddings[s],entity_embeddings[o],c),0)
                        concat_embs.append(concat_emb)
            if(len(localized_context) == 0):
                continue
            localized_context = torch.stack(localized_context)
            embs = torch.stack(concat_embs)
            
            triggers = list(set(triggers))
            # ---------- Pairwise Comparisons and Predictions ------------

            scores = torch.matmul(embs,self.relation_embeddings.T)
            nota_scores = torch.matmul(embs,self.nota_embeddings.T)
            nota_scores = nota_scores.max(dim=-1,keepdim=True)[0]
            scores = torch.cat((nota_scores, scores), dim=-1)
            predictions = torch.argmax(scores, dim=-1, keepdim=False)
            #Achtung: NOTA wird an 0. Stelle gesetzt
            
            if self.training:
            # ---------- ATLoss with one-hot encoding for true labels ------------
                targets = []
                for r in relation_candidates:
                    onehot = torch.zeros(len(relation_types))
                    if r in relation_labels[batch_i]:
                        onehot[relation_labels[batch_i][r]] = 1.0
                    targets.append(onehot)
                targets = torch.stack(targets).to(self.model.device)
                loss += self.at_loss(scores,targets)
                counter += 1
            
            # ---------- Inference ------------
            triples = []
            for idx,pair in enumerate(relation_candidates):
                triple = {
                    pair:relation_types[predictions[idx]]
                }
                triples.append(triple)
            batch_triples.append(triples)
                
            events = []
            for trigger in triggers:
                trigger_word = entity_types[batch_i][trigger]
                event_type = trigger_word.split(".TRIGGER")[0]
                trigger_span = entity_spans[batch_i][trigger][0]
                t_start = trigger_span[0]
                t_end = trigger_span[1]
                arguments = []
                #print(f"event_type: {event_type}, feasible roles: {self.feasible_roles[event_type]}")
                for t in triples:
                    #Triples ist Liste von dicts, gibt es eine geschicktere Möglichkeit an s,o,r zu kommen?
                    for i in t.items():
                        s = i[0][0]
                        o = i[0][1]
                        r = i[1] 
                        
                        if s == trigger:
                            if r in self.feasible_roles[event_type]:
                                a_span = entity_spans[batch_i][o][0]
                                a_start = a_span[0]
                                a_end = a_span[1]
                                argument = {
                                    
                                    'role':r,
                                    'start':a_start,
                                    'end':a_end,
                                    'text':batch_text[batch_i][a_start:a_end][0]
                                }
                                arguments.append(argument)
                
                event = {
                    'event_type':event_type,
                    'trigger': {'start':t_start ,'end':t_end, 'text':batch_text[batch_i][t_start:t_end][0]},
                    'arguments':arguments
                }
                events.append(event)
            batch_events.append(events)
        if(counter == 0):
                return torch.autograd.Variable(loss,requires_grad=True)
        else:
            return (mention_loss+loss)/counter, batch_triples, batch_events


