from dataclasses import dataclass, field
import os
import json
import mlflow
from torch import nn
from torch import tensor
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.parallel import DataParallel
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data.dataset import Dataset, IterableDataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from optuna import Trial

from transformers import BertForSequenceClassification, EvalPrediction, MegatronBertForTokenClassification
from transformers.integrations import MLflowCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers.trainer import Trainer
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from typing import Any, Dict, Optional, List, Union
import logging
import torch
import numpy as np
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from transformers.training_args import TrainingArguments
import settings
from utils import preprocess_data as pr
from utils import detokenize
from utils import conlleval
from utils.logger_utils import get_custom_logger
from utils import file_utils

logger = get_custom_logger(__name__)


arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

class AppMLflowCallback(MLflowCallback):

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        # will call initalization again
        self.on_train_begin(args, state, control, **kwargs)
        self.on_log(args, state, control, logs=metrics, model=model, **kwargs)
        self.on_train_end(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        if self._initialized and state.is_world_process_zero:
            self._initialized = False

class MegatronBertForTokenClassificationMod(MegatronBertForTokenClassification):

    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob_classification_layer)

        # Initialize weights and apply final processing
        self.post_init()

@dataclass
class NerTrainingArguments(TrainingArguments):
    """
    NER Training Arguments

    Args:
    """
    experiment_name: str = field(
        default = "BioBERT NER training.",
        metadata={"help": "Experiment name."}        
    )
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )
    output_dir: str = field(
        default = settings.TMP_DIR + "/output",
        metadata = TrainingArguments.__dataclass_fields__["output_dir"].metadata
    )
    do_hp_search: bool = field(default=False, metadata={"help": "Whether to run hyperparameter search."})
    
    calculate_outer_fold_results: bool = field(default=False, metadata={"help": "Calculate nested CVs outer fold results"})
    o_fold: Optional[int] = field(default=-1, metadata={"help": "Outer fold of nested CV."})
    i_trial: Optional[int] = field(default=-1, metadata={"help": "Trial of nested CV."})
    i_fold: Optional[int] = field(default=-1, metadata={"help": "Inner fold of nested CV."})

class NerTrainer(Trainer):
    """
    Keeps everything just like in the original huggingface Trainer
    If needed, modify it.

    Args:
        Trainer ([Trainer]): Huggingface Trainer
    """

    def __init__(self, *args, **kwargs):
      self.test_dataset = kwargs.pop('test_dataset')
      self.ner_args: NerTrainingArguments = kwargs.get("args") # type:ignore
      super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
#    def prediction_loop(
#        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
#    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, extends :obj:`Trainer.prediction_loop()` to add entity wise metrics.
        """

        result = super().evaluation_loop(dataloader,description,prediction_loss_only)
        result = self._compute_entity_wise_metrics(dataloader, result)
        return result
    
    # def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    #     inputs = super()._prepare_inputs(inputs)
    #     inputs.pop('id', None)        
    #     return inputs

    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

    def _compute_entity_wise_metrics(self, dataloader: DataLoader, prediction_output: EvalLoopOutput) -> EvalLoopOutput:
        """
        Compute entity wise metrics.
        """
        #logger.info("TYPE %s", type(dataloader.dataset))
        # first_index = dataloader.dataset.indices[0]
        # input_ids = dataloader.dataset.dataset.features[first_index].input_ids
        # predicted_label_classes = prediction_output.label_ids[0]
        # predicted_labels = [self.model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist() if id in self.model.config.id2label]
        # for id, label in zip(input_ids, predicted_labels):
        #     print(self.tokenizer.decode([id]), label)
            
        true_predictions, true_labels = align_predictions(prediction_output.predictions, prediction_output.label_ids, self.model.config.id2label) #type: ignore

        assert self.state.epoch is not None
        epoch = int(self.state.epoch) # self.state.global_step
        report = classification_report(true_labels, true_predictions)
        # with open(self.args.output_dir + "/" + 'classification_report_token_wise__epoch_' + str(epoch) + '.txt', 'w') as file:
        #    file.write(report)
        mlflow.log_text(report, 'classification_report_token_wise__epoch_' + str(epoch) + '.txt')

        predicted_labels = detokenize.preprocess_labels(true_predictions)
        valid_labels = detokenize.preprocess_labels(true_labels)
        # from torch.utils.data.dataset import Subset
        
        # if isinstance(dataloader.dataset, Subset) and isinstance(dataloader.dataset.dataset, Subset):
        #     to_use_dataset = dataloader.dataset.dataset.dataset
        # if isinstance(dataloader.dataset, Subset) and not isinstance(dataloader.dataset.dataset, Subset):
        #     to_use_dataset = dataloader.dataset.dataset
        # else:
        #     to_use_dataset = dataloader.dataset

        return_tokens = [instance["return_tokens"] for instance in dataloader.dataset]

        val_tokens = detokenize.preprocess_tokens(return_tokens) #type: ignore
        if "features" in dataloader.dataset[0]:
            features = [instance["features"] for instance in dataloader.dataset]
            val_offsets = detokenize.extract_offsets(features) #type: ignore
        elif "offsets" in dataloader.dataset[0]:
            offsets = [instance["offsets"] for instance in dataloader.dataset]
            val_offsets = detokenize.extract_offsets_list(offsets) #type: ignore
        else:
            raise Exception("offsets couldn't be found.")
        
        output_lines = detokenize.main(predicted_labels, valid_labels, val_tokens, val_offsets, "test") # TODO: mode
        counts = conlleval.evaluate(output_lines)
        _result = conlleval.return_result(counts)

        # file_utils.write_lines(self.args.output_dir, "pred_labels__epoch_" + str(epoch) + ".txt", predicted_labels)
        # file_utils.write_lines(self.args.output_dir, "valid_labels__epoch_" + str(epoch) + ".txt", valid_labels)
        # file_utils.write_lines(self.args.output_dir, "val_tokens__epoch_" + str(epoch) + ".txt", val_tokens)
        # file_utils.write_lines(self.args.output_dir, "val_offsets__epoch_" + str(epoch) + ".txt", val_offsets)
        # file_utils.write_lines(self.args.output_dir, "output_lines__epoch_" + str(epoch) + ".txt", output_lines)
        
        mlflow.log_text("\n".join(predicted_labels),  "pred_labels__epoch_" + str(epoch) + ".txt")
        mlflow.log_text("\n".join(valid_labels),  "valid_labels__epoch_" + str(epoch) + ".txt")
        mlflow.log_text("\n".join(val_tokens), "val_tokens__epoch_" + str(epoch) + ".txt")
        mlflow.log_text("\n".join(val_offsets), "val_offsets__epoch_" + str(epoch) + ".txt")
        mlflow.log_text("\n".join(output_lines), "output_lines__epoch_" + str(epoch) + ".txt")
        mlflow.log_text(counts.toJSON(), 'counts__epoch_' + str(epoch) + '.json')
        mlflow.log_text(json.dumps(_result, indent = 4), 'result__epoch_' + str(epoch) + '.json')
        
        # with open(self.args.output_dir + "/" + 'counts__epoch_' + str(epoch) + '.json', 'w') as json_file:
        #    json_file.write(counts.toJSON())
        # with open(self.args.output_dir + "/" + 'result__epoch_' + str(epoch) + '.json', 'w') as json_file:
        #    json.dump(_result, json_file)

        for class_type in _result["accuracy"]:
            prediction_output.metrics[f"eval_entity_wise_accuracy_" + class_type] = (_result["accuracy"][class_type]/ 100.0)
        for class_type in _result["precision"]:
            prediction_output.metrics[f"eval_entity_wise_precision_" + class_type] = (_result["precision"][class_type]/ 100.0)
        for class_type in _result["recall"]:
            prediction_output.metrics[f"eval_entity_wise_recall_" + class_type] = (_result["recall"][class_type]/ 100.0)
        for class_type in _result["f1"]:
            prediction_output.metrics[f"eval_entity_wise_f1_" + class_type] = (_result["f1"][class_type]/ 100.0)

        return prediction_output

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict[int, str]):
    """
    Align predictions.
    """
    
    # label_map: Dict[int, str] = {i: label for i, label in enumerate(pr.get_labels())}
    
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    true_labels = [[] for _ in range(batch_size)]
    predicted_labels = [[] for _ in range(batch_size)]

    # take only predictions that are not padded.
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100 and label_ids[i, j] != -72056494526300260: # TODO: move padding id to somewhere else
                true_labels[i].append(label_map[label_ids[i][j]])
                predicted_labels[i].append(label_map[preds[i][j]])

    return predicted_labels, true_labels

def compute_additional_metrics(preds_labels: EvalPrediction) -> Dict:
    """
    In addition to the loss returned in the trainer, also return a dict with
    f1_score, accuracy, precision and recall at evaluation time.

    Args:
      preds_labels (EvalPrediction): Predicted labels

    Returns:
        Dict: Metrics dict with accuracy, balanced accuracy, f1-score, precision, recall
    """
    predicted_labels, true_labels = align_predictions(preds_labels.predictions, preds_labels.label_ids, settings.LABEL_MAP) #type: ignore
    
    metrics_dict = {
        "token_wise_accuracy": accuracy_score(true_labels, predicted_labels),
        "token_wise_precision": precision_score(true_labels, predicted_labels),
        "token_wise_recall": recall_score(true_labels, predicted_labels),
        "token_wise_f1": f1_score(true_labels, predicted_labels),
    }
    # logger.info("Classification Report: %s", classification_report(true_labels, predicted_labels))
    # logger.info("Metrics dictionary: %s", metrics_dict)
    
    return metrics_dict


class BC5CDRDataset(Dataset):
    """ 
    Helper class to wrap the bc5cdr data into a format that can be easily processed
    by pytorch/the transformers library.
    """

    @staticmethod
    def get_data(file):
        data_file = os.path.join(file)

        # gather training examples:
        label_list = pr.get_labels()
        #num_labels = len(label_list)
        sentences, labels, offsets = pr.extract_sent_labels(data_file, 'train') # TODO: refactor and remove mode
        instances = pr.create_examples(sentences, labels, offsets, 'train') # TODO: refactor and remove mode
        return BC5CDRDataset(instances)
    
    def __init__(self, instances: List):
        self.instances = instances
        #self.num_labels = len(.label_encoding)
        #self.max_chunks = np.max([len(i.tokens) for i in self.instances])

    def set_features(self, label_list, max_seq_length, tokenizer, batch_size, mode):
        (train_features,return_tokens) = pr.convert_examples_to_features(self.instances, label_list, max_seq_length, tokenizer, mode)
        self.mode = mode
        self.features = train_features
        self.return_tokens = return_tokens
        self.all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_ids for f in self.features], dtype=torch.long)
        self.tensor_dataset = TensorDataset(self.all_input_ids, self.all_attention_mask, self.all_segment_ids, self.all_label_ids)
        self.sampler = RandomSampler(self.tensor_dataset)
        self.dataloader = DataLoader(self.tensor_dataset, sampler=self.sampler, batch_size=batch_size)

    def get_features (self):
        return self.features

    def __getitem__(self, idx):
        item = dict()
        item['input_ids'] = tensor(self.features[idx].input_ids, dtype=torch.long)
        item['attention_mask'] = tensor(self.features[idx].input_mask, dtype=torch.long)
        item['token_type_ids'] = tensor(self.features[idx].segment_ids, dtype=torch.long)
        item['label_ids'] = tensor(self.features[idx].label_ids, dtype=torch.long)
        item['features'] = self.features[idx]
        item['return_tokens'] = self.return_tokens[idx]
        
        return item

    def __len__(self):
        return len(self.features)

    def getTensorDataset(self):
        return self.tensor_dataset

    def getRandomSampler(self):
        return self.sampler
    
    def getDataLoader(self):
        return self.dataloader


@dataclass
class NerModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    #freeze_encoder: bool = field(default=False, metadata={"help": "Whether to freeze the encoder."})
    #freeze_embeds: bool = field(default=False, metadata={"help": "Whether to freeze the embeddings."})


@dataclass
class NerDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=settings.DATA_DIR,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    train_dataset_file_name: str = field(
        default = "train.tsv",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    eval_dataset_file_name: str = field(
        default = "devel.tsv",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    test_dataset_file_name: str = field(
        default = "test.tsv",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})

@dataclass
class MirnaDataTrainingArguments(NerDataTrainingArguments):

    data_dir: str = field(
        default = settings.MIRNA_DATA_DIR,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )

def hyperparameter_space(trial: Trial) -> Dict[str, float]:
    hps =  {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_loguniform('weight_decay', 1e-10, 1e-1),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "seed": trial.suggest_int("seed", 20, 40, 4),
    }
    return hps

def ner_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The compute objective to maximize/minimize when doing an hyperparameter search. It is the f1-score.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    assert "eval_entity_wise_f1" in metrics, "Key eval_entity_wise_f1 should exists in metrics"
    f1_score = metrics.get("eval_entity_wise_f1")
    return f1_score if isinstance(f1_score, float) else -1.0

from optuna.study import Study
import optuna
def create_or_get_optuna_study(o_fold: int = None, study_name_prefix = "disease-entity", extended_testset: str = None, storage = settings.OTPUNA_STORAGE, direction: str = "maximize") -> Study:
    """
    Create or get optuna study for the specific outer and inner fold

    Args:
        o_fold (int): Outer fold
        study_name_prefix (str): study name prefix
        extended_testset (str): name of extended testset
        storage (str): Optuna storage

    Returns:
        [Study]: Optuna study that was created or loaded 
    """
    if o_fold is not None:
        study_name = study_name_prefix + "-ofold-" + str(o_fold)
    else:
        study_name = study_name_prefix
    
    if extended_testset is not None:
        study_name = study_name + "_" + extended_testset
        
    study: Study = optuna.create_study(
        study_name=study_name,
        storage = storage,
        direction = direction,
        pruner=optuna.pruners.HyperbandPruner(), 
        load_if_exists=True
    )

    return study
