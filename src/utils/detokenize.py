import os
import json
from typing import List
from deprecated import deprecated

from utils import detokenize_old

@deprecated(version='', reason="This method is deprecated, use preprocess_results()")
def read_token(pred_labels, valid_labels, tokens, offsets):
    
    if valid_labels != None:
        d = {'token': [], 'pred': [], 'valid': [], 'offset':[]}
    else:
        d = {'token': [], 'pred': [], 'offset':[]}
        
    with open(pred_labels) as f:
        for line in f:
            d['pred'].append(line[:line.find('\n')])
    
    if valid_labels != None:
        with open(valid_labels) as f:
            for line in f:
                d['valid'].append(line[:line.find('\n')])

    with open(tokens) as f:
        for line in f:
            if line != '[SEP]\n':
                d['token'].append(line[:line.find('\n')])
    
    with open(offsets) as f:
        for line in f:
            if line != '[SEP]\n':
                d['offset'].append(line[:line.find('\n')])
    
    return d

def preprocess_results(pred_labels, valid_labels, tokens, offsets):
    
    result = {}
    
    # remove [CLS] and [SEP] tags
    result["token"] = [[token for token in entries if token not in ["[SEP]", "[CLS]"]] for entries in tokens]
    result["offset"] = [[offset for offset in entries if offset not in ["[SEP]", "[CLS]"]] for entries in offsets]
    result['pred'] = [[label for label in entries if label not in ["[CLS]"]] for entries in pred_labels]
    if valid_labels != None:
        result['valid'] = [[label for label in entries if label not in ["[CLS]"]] for entries in valid_labels]

    return result

def detokenize(data):
        
    # initialize
    ner = {}
    ner['words'] = []
    ner['pred_labels'] = []
    if 'valid' in data.keys():
        ner['valid_labels'] = []
    ner['offsets'] = []
    
    # store
    tokens = data['token']
    pred_labels = data['pred']
    valid_labels = None
    if 'valid' in data.keys():
        valid_labels = data['valid']
    offsets = data['offset']
    
    # transform tokens
    for i, entry in enumerate(tokens):
        sent_tokens = []
        sent_token_offsets = []
        sent_token_pred_labels = []
        sent_token_valid_labels = []
        
        for j, token in enumerate(entry):
            if token[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
                sent_tokens[-1] = sent_tokens[-1] + token[2:]  # append pieces
            else:
                sent_tokens.append(token)
                sent_token_pred_labels.append(pred_labels[i][j])
                if valid_labels != None: # 'valid' in data.keys():
                    sent_token_valid_labels.append(valid_labels[i][j])
                sent_token_offsets.append(offsets[i][j])
        
        ner['words'].append(sent_tokens)
        ner['pred_labels'].append(sent_token_pred_labels)
        if 'valid' in data.keys():
            ner['valid_labels'].append(sent_token_valid_labels)
        ner['offsets'].append(sent_token_offsets)
        
    return ner

@deprecated(version='', reason="This method is deprecated. It uses files mode, that is not available. Better refactor.")
def main(pred_labels, valid_labels, tokens, offsets, mode):
    data = {'token': tokens, 'pred': pred_labels, 'valid': valid_labels, 'offset': offsets}

    if mode == 'predict':
        data["valid_labels"] = None

    ner_data = detokenize_old.detokenize(data)

    output_lines = []
    for i in range(1, len(ner_data['words'])):  # skip first row because it is only [CLS]
        if ner_data['words'][i] == '[CLS]':
            output_lines.append('[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]')
        else:
            if 'valid_labels' in ner_data.keys():  
                output_lines.append("{0}\t{1}\t{2}\t{3}".format(
                    ner_data['words'][i], ner_data['valid_labels'][i], ner_data['pred_labels'][i], ner_data['offsets'][i]))
            else:
                output_lines.append("{0}\t{1}\t{2}".format(
                    ner_data['words'][i], ner_data['pred_labels'][i], ner_data['offsets'][i]))
    output_lines.append('[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]'+ '\t' + '[CLS]')
    return output_lines

def preprocess_tokens(instances):
    new_tokens_flattened = []
    for tokens in instances:
        for token in tokens:
            if token != "**NULL**" and not token.startswith('[SEP]'):
                new_tokens_flattened.append(token)
    return new_tokens_flattened

def extract_offsets(features):
    offsets = [f.offset for f in features] # list of lists
    new_offsets = []
    for i in offsets:
        j: str
        for j in i:
            if j != "**NULL**" and not j.startswith('[SEP]'):
                new_offsets.append(j)
    return new_offsets

def extract_offsets_list(offsets):
    new_offsets = []
    for i in offsets:
        j: str
        for j in i:
            if j != "**NULL**" and not j.startswith('[SEP]'):
                new_offsets.append(j)
    return new_offsets

def preprocess_labels(instances):
    new_labels = []
    for instance in instances:
        i = 0
        for label in instance:
            i +=1
            if label.startswith('[SEP]') and i == len(instance): 
                continue
            
            # if separator occures somewhere in the sentence replace it with O
            if label.startswith('[SEP]'):
                label = "O"
            
            new_labels.append(label)
    return new_labels

# if __name__ == '__main__':

