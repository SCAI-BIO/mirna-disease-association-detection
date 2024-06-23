import pandas as pd
import os
import json

import settings


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, offset, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid  # gives an id for the whole sentence
        self.text_a = text_a  # whole sentence (untokenized!)
        self.text_b = text_b # not needed for NER 
        self.label = label  # labels for whole untokenized sentence
        # add offset: 
        self.offset = offset 

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, offset):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # add offset information 
        self.offset = offset


def load_data(file_path):
    data = pd.read_csv(file_path, encoding="latin1", sep="\t", skip_blank_lines=False)
    data["Tag"].fillna("empty", inplace=True)  # empty row = end of sentence, need identifier for this
    return data


def extract_sent_labels(data_path, mode):
    file_ = open(data_path, 'r')
    lines = file_.readlines()
    file_.close()
    sentence_list = []
    labels_list = []
    offset_list = []
    temp_sent = []
    temp_labels = []
    temp_offsets = []
    for line in lines[1:]:
        splitted = line.split('\t') 
        if line == '\n':  # = end of sentence
            sentence_list.append(temp_sent)
            labels_list.append(temp_labels)
            offset_list.append(temp_offsets)
            temp_sent = []
            temp_labels = []
            temp_offsets = []
        else:
            if mode == 'predict':
                o = splitted[1][:splitted[1].find('\n')]
                t = None
            else:
                t = splitted[1]
                o = splitted[2][:splitted[2].find('\n')]
            temp_labels.append(str(t))
            w = splitted[0]
            temp_sent.append(str(w))
            temp_offsets.append(str(o))  
            
    sentences = [" ".join([s for s in sent]) for sent in sentence_list]
    labels = [" ".join([l for l in label]) for label in labels_list]
    offsets = [" ".join([i for i in offset]) for offset in offset_list]
    return sentences, labels, offsets


def get_labels():
    # return settings.LABEL_MAP
    return ["B", "I", "O", "X", "[CLS]", "[SEP]"]


def create_examples(sentences, labels, offsets, set_type):  # get train_examples = _create_example(read_data, 'train')
    examples = []
    for i in range(len(sentences)):
        guid = "%s-%s" % (set_type, i)
        text = sentences[i]
        label = labels[i]
        offset = offsets[i]
        examples.append(InputExample(guid=guid, text_a=text, text_b=None, offset = offset, label=label))
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    features = []
    return_tokens = []

    label_map = {label: i for i, label in enumerate(label_list)}

    for example in examples:

        #textlist = example.text_a.split(' ')
        #labellist = example.label.split(' ')
        #offsetlist = example.offset.split(' ')
        
        textlist = example.text_a.split()
        labellist = example.label.split()
        offsetlist = example.offset.split()

        tokens = []
        labels = []
        offsets = []

        tokens.append('[CLS]')
        labels.append('[CLS]')
        offsets.append('[CLS]')
        # print(textlist)
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
        #    print(token)
            tokens.append(token[0])
            labels.append(labellist[i])
            offsets.append(offsetlist[i])
            if len(token) > 1:
                for c in range(1, len(token)):
                    labels.append('X')
                    offsets.append('X')
                    tokens.append(token[c])

        # check for restriction of length of sentence: 128
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[:max_seq_length - 1]  # -1 because 'SEP' must be added
            labels = labels[:max_seq_length - 1]
            offsets = offsets[:max_seq_length-1]

        tokens.append('[SEP]')
        labels.append('[SEP]')
        offsets.append('[SEP]')

        input_mask = [1] * len(labels)
        segment_ids = [0] * len(labels)

        # convert tokens and labels to corresponding ids:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if mode != "predict":
            label_ids = [label_map[i] for i in labels]
        else:
            label_ids = [0 for i in range(len(labels))]

        while len(input_ids) < max_seq_length:
            input_ids.append(0)  # padding
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-100)
            tokens.append("**NULL**")
            offsets.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(offsets) == max_seq_length

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids, 
            offset = offsets))

        #if mode != 'train':
        example_tokens = []
        for token in tokens:
            if token != "**NULL**":
                example_tokens.append(token)
        return_tokens.append(example_tokens)

    return (features, return_tokens)

