import logging
import os
import csv
from typing import List, cast
import numpy as np
from deprecated.classic import deprecated
from transformers.models.bert.tokenization_bert import BertTokenizer

from utils.logger_utils import get_custom_logger

logger = get_custom_logger(__name__, level=logging.INFO)

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

@deprecated()
def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    return lines

def get_examples(data_file, mode): # mode == "train", "test" or "dev"
    # lines = read_tsv(os.path.join(data_dir, mode+".tsv"))
    lines = read_tsv(data_file)
    entries = []
    for (i, line) in enumerate(lines):
        entry = {}
        entry["text_a"] = line[0]
        entry['label'] = line[1] if mode != "predict" else "0"
        entries.append(entry)
    return create_examples(entries, mode)

def create_examples(entries, classification_mode) -> List[InputExample]:
    """Creates examples for the training and dev sets."""
    examples = []
    i = 0
    for entry in entries:
        guid = "%s-%s" % (classification_mode, i)
        i = i+1
        text_a = entry['text_a']
        label = entry['label'] if classification_mode != "predict" else "0"
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer: BertTokenizer, mode) -> List[InputFeatures]:
    features: List[InputFeatures] = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Showing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        features.append(feature)

    return features

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode) -> InputFeatures:
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(input_ids=[0] * max_seq_length,
                             input_mask=[0] * max_seq_length,
                             segment_ids=[0] * max_seq_length,
                             label_id=0,
                             is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

    _input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if not isinstance(_input_ids, List):
        logger.error("token input ids should be a list")
        raise AssertionError()

    input_ids = cast(List[int], _input_ids)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length, "length of input_ids should be equal to max_seq_length"
    assert len(input_mask) == max_seq_length, "length of input_ids should be equal to max_seq_length"
    assert len(segment_ids) == max_seq_length, "length of input_ids should be equal to max_seq_length"
    
    if mode == 'predict':
        label_id = None
    else:
        label_id = label_map[example.label]
        
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

@deprecated(version='', reason="This method is deprecated. Probably don't need to call there is no tokens_b.")
def _truncate_seq_pair(tokens_a, tokens_b, max_length):  # I think that I don't need this one because
    # I don't have "tokens_b"
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_labels():
    """See base class."""
    return ["0", "1"]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
