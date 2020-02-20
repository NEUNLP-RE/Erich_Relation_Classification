from __future__ import absolute_import, division, print_function

import logging
import os
import codecs

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, e1_pos, e2_pos, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            e1_pos: list. The position of entity1.
            e2_pos: list. The position of entity2.
            text: string. The text need to parse.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, e1_pos, e2_pos, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SemevalProcessor(object):
    """Processor for the Semeval data set."""

    def get_examples(self, data_dir, mode):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(data_dir, mode + ".txt")), mode)

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with codecs.open(os.path.join(data_dir, "relation2id.txt"), "r", "utf-8") as fr:
            for l in fr:
                rel, ids = l.strip().split("\t")
                labels.append(ids)
        return labels

    @classmethod
    def _read_data(cls, input_file):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            tokens = line.split()
            e1_pos = [int(i) for i in tokens[1:3]]
            e2_pos = [int(i) for i in tokens[3:5]]
            text = " ".join(tokens[5:])
            label = tokens[0]
            examples.append(
                InputExample(guid=guid, e1_pos=e1_pos, e2_pos=e2_pos, text=text, label=label))
        return examples


def tokenize_example_text(sample, tokenizer):
    words = sample.text.split()
    e1_p, e2_p = sample.e1_pos, sample.e2_pos
    # text_spans = [" ".join(words[:e1_p[0]] + ["<e1>"]), " ".join(words[e1_p[0]:e1_p[1] + 1]),
    #               " ".join(["</e1>"] + words[e1_p[1] + 1:e2_p[0]] + ["<e2>"]),
    #               " ".join(words[e2_p[0]:e2_p[1] + 1]), " ".join(["</e2>"] + words[e2_p[1] + 1:])]
    text_spans = [" ".join(words[:e1_p[0]] + ["$"]), " ".join(words[e1_p[0]:e1_p[1] + 1]),
                  " ".join(["$"] + words[e1_p[1] + 1:e2_p[0]] + ["#"]),
                  " ".join(words[e2_p[0]:e2_p[1] + 1]), " ".join(["#"] + words[e2_p[1] + 1:])]
    text_spans = [tokenizer.tokenize(s) for s in text_spans]
    len_spans = [len(s) for s in text_spans]
    sample.text = sum(tuple(text_spans), [])
    sample.e1_pos = [sum(len_spans[:1]), sum(len_spans[:2]) - 1]
    sample.e2_pos = [sum(len_spans[:3]), sum(len_spans[:4]) - 1]
    return sample


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        token_example = tokenize_example_text(example, tokenizer)
        tokens = token_example.text

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            token_example.e1_pos = [i + 1 for i in token_example.e1_pos]
            token_example.e2_pos = [i + 1 for i in token_example.e2_pos]
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("e1_pos: %s" % " ".join([str(i) for i in token_example.e1_pos]))
            logger.info("e2_pos: %s" % " ".join([str(i) for i in token_example.e2_pos]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          e1_pos=token_example.e1_pos,
                          e2_pos=token_example.e2_pos,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


processors = {
    "semeval": SemevalProcessor,
}

output_modes = {
    "semeval": "classification",
}
