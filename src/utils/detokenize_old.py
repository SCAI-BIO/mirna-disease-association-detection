import os
from deprecated import deprecated

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


def detokenize(d):
    if 'valid' in d.keys():
      
        ner = {'words': [], 'pred_labels': [], 'valid_labels': [], 'offsets': []}

        for t, p, v, o in zip(d['token'], d['pred'], d['valid'], d['offset']):
            if t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
                ner['words'][-1] = ner['words'][-1] + t[2:]  # append pieces
            else:
                ner['words'].append(t)
                ner['pred_labels'].append(p)
                ner['valid_labels'].append(v)
                ner['offsets'].append(o)
    else: 
      
        ner = {'words': [], 'pred_labels': [], 'offsets': []}

        for t, p, o in zip(d['token'], d['pred'],  d['offset']):
            if t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
                ner['words'][-1] = ner['words'][-1] + t[2:]  # append pieces
            else:
                ner['words'].append(t)
                ner['pred_labels'].append(p)
                ner['offsets'].append(o)        
    return ner

@deprecated(version='', reason="This method is deprecated. It uses files mode, that is not available. Better refactor.")
def main(dir_, mode):

    pred = os.path.join(dir_, "pred_labels.txt")
    valid = os.path.join(dir_, "valid_labels.txt")
    token = os.path.join(dir_, "token.txt")
    offset = os.path.join(dir_, "offsets.txt")
    output = os.path.join(dir_, "result_entity.tsv")
    if mode != 'predict':
        data = read_token(pred, valid, token, offset)
    else:
        data = read_token(pred, None, token, offset)
    ner_data = detokenize(data)

    with open(output, 'w') as f:
        for i in range(1, len(ner_data['words'])):  # skip first row because it is only [CLS]
            if ner_data['words'][i] == '[CLS]':
                f.write('[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]' + '\n')
            else:
                if 'valid_labels' in ner_data.keys():  
                    f.write("%s\t%s\t%s\t%s\n" % (
                    ner_data['words'][i], ner_data['valid_labels'][i], ner_data['pred_labels'][i], ner_data['offsets'][i]))
                else:
                    f.write("%s\t%s\t%s\n" % (
                    ner_data['words'][i], ner_data['pred_labels'][i], ner_data['offsets'][i]))
        f.write('[CLS]' + '\t' + '[CLS]' + '\t' + '[CLS]'+ '\t' + '[CLS]')
