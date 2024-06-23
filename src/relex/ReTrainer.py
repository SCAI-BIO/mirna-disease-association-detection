from dataclasses import dataclass, field
import os
import json
import plotly.express as px

from torch import tensor
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from optuna import Trial

# from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput, EvalPrediction

from sklearn.metrics import f1_score, precision_score, recall_score

from typing import Dict, Optional, List
import logging
import torch
import mlflow
import numpy as np
from transformers.trainer_utils import PredictionOutput
from transformers.training_args import TrainingArguments
import settings
#from utils import detokenize
#from utils import conlleval
from utils import logger_utils
#from utils import file_utils
from nerex.NerTrainer import arg_to_scheduler
import relex.preprocessing as pr
from torchmetrics import (AUROC, F1Score, ROC, Accuracy, AveragePrecision, 
                          Precision, PrecisionRecallCurve,
                          Recall, ConfusionMatrix, MetricCollection)


logger = logger_utils.get_custom_logger(__name__, level=logging.INFO)

@dataclass
class ReTrainingArguments(TrainingArguments):
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


class ReTrainer(Trainer):
    """
    Keeps everything just like in the original huggingface Trainer
    If needed, modify it.

    Args:
        Trainer ([Trainer]): Huggingface Trainer
    """

    def __init__(self, *args, **kwargs):
        self.test_dataset = kwargs.pop('test_dataset')
        self.re_args: ReTrainingArguments = kwargs.get("args") # type:ignore
        self.valid_metrics = MetricCollection([
            # Accuracy(task="multiclass", num_classes=2, average=None), 
            Precision(task="multiclass", num_classes=2, average=None), 
            Recall(task="multiclass", num_classes=2, average=None), 
            F1Score(task="multiclass", num_classes=2, average=None),
            AveragePrecision(task="multiclass", num_classes=2, average=None),
            ConfusionMatrix(task="multiclass", num_classes=2, average=None),
            PrecisionRecallCurve(task="multiclass", num_classes=2, average=None),
            AUROC(task="multiclass", num_classes=2, average=None),
            ROC(task="multiclass", num_classes=2, average=None),
        ], prefix='eval_')
# >>> metrics = MetricCollection([
# ...     MetricCollection([
# ...         MulticlassAccuracy(num_classes=3, average='macro'),
# ...         MulticlassPrecision(num_classes=3, average='macro')
# ...     ], postfix='_macro'),
# ...     MetricCollection([
# ...         MulticlassAccuracy(num_classes=3, average='micro'),
# ...         MulticlassPrecision(num_classes=3, average='micro')
# ...     ], postfix='_micro'),
# ... ], prefix='valmetrics/')
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, extends :obj:`Trainer.prediction_loop()` to add entity wise metrics.
        """

        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        result = self._compute_additional_metrics(dataloader, result)
        return result

    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)

    def _compute_additional_metrics(self, dataloader: DataLoader, prediction_output: EvalLoopOutput) -> EvalLoopOutput:
        
        assert self.state.epoch is not None
        epoch = int(self.state.epoch) # self.state.global_step
        
        # report = classification_report(prediction_output.label_ids, prediction_output.predictions)
        # mlflow.log_text(report, 'classification_report_token_wise__epoch_' + str(epoch) + '.txt')

        predictions = torch.nn.Softmax(dim=1)(torch.tensor(prediction_output.predictions))
        label_ids = torch.tensor(prediction_output.label_ids)

        self.valid_metrics.update(predictions, label_ids)
        result = self.valid_metrics.compute()
        self.valid_metrics.reset()
        
        assert self.valid_metrics.prefix is not None
        
        self.log_roc_graph(result, self.valid_metrics.prefix + "Multiclass")
        self.log_prc_graph(result, self.valid_metrics.prefix + "Multiclass")

        result.pop(self.valid_metrics.prefix + 'MulticlassROC', None)
        result.pop(self.valid_metrics.prefix + 'MulticlassPrecisionRecallCurve', None)
        f1_score = result.pop(self.valid_metrics.prefix + 'MulticlassF1Score', torch.Tensor([-1,-1]))
        result[self.valid_metrics.prefix + "F1_class_0"] = f1_score[0].item()
        result[self.valid_metrics.prefix + "F1_class_1"] = f1_score[1].item()
        precision = result.pop(self.valid_metrics.prefix + 'MulticlassPrecision', torch.Tensor([-1,-1]))
        result[self.valid_metrics.prefix + "Precision_class_0"] = precision[0].item()
        result[self.valid_metrics.prefix + "Precision_class_1"] = precision[1].item()
        recall = result.pop(self.valid_metrics.prefix + 'MulticlassRecall', torch.Tensor([-1,-1]))
        result[self.valid_metrics.prefix + "Recall_class_0"] = recall[0].item()
        result[self.valid_metrics.prefix + "Recall_class_1"] = recall[1].item()
        auroc = result.pop(self.valid_metrics.prefix + 'MulticlassAUROC', torch.Tensor([-1,-1]))
        result[self.valid_metrics.prefix + "AUROC"] = auroc[0].item()
        average_precision = result.pop(self.valid_metrics.prefix + 'MulticlassAveragePrecision', torch.Tensor([-1,-1]))
        result[self.valid_metrics.prefix + "average_precision_class_0"] = average_precision[0].item()
        result[self.valid_metrics.prefix + "average_precision_class_1"] = average_precision[1].item()
        confmat = result.pop(self.valid_metrics.prefix + 'MulticlassConfusionMatrix', torch.Tensor([[-1,-1],[-1,-1]]))
        result[self.valid_metrics.prefix + "ConfMat_TN"] = confmat[0][0].item()
        result[self.valid_metrics.prefix + "ConfMat_FN"] = confmat[0][1].item()
        result[self.valid_metrics.prefix + "ConfMat_FP"] = confmat[1][0].item()
        result[self.valid_metrics.prefix + "ConfMat_TP"] = confmat[1][1].item()


        assert prediction_output.metrics is not None
        prediction_output.metrics.update(result)
        
        mlflow.log_metrics(result)

        return prediction_output

    def log_roc_graph(self, result, mode = "eval_") -> None:
        fpr, tpr, thresholds = result[mode + "ROC"] # self.val_metric_roc.compute()

        fpr = fpr.cpu().numpy() if isinstance(fpr, torch.Tensor) else fpr
        tpr = tpr.cpu().numpy() if isinstance(tpr, torch.Tensor) else tpr
        thresholds = thresholds.cpu().numpy() if isinstance(thresholds, torch.Tensor) else thresholds
        # self.local_logger.info("AUROC type: %s", type(result[mode + "AUROC"]))
        # self.local_logger.info("AUROC len: %s", len(result[mode + "AUROC"]))
        # self.local_logger.info("AUROC: %s", result[mode + "AUROC"])
        fig = px.area(
            x=fpr[1], y=tpr[1],
#            title=f'ROC Curve (AUC={self.val_metric_auroc.compute():.4f})',
            # title=f'ROC Curve (AUROC={result[mode + "AUROC"]:.4f})',
            title=f'ROC Curve (AUROC-0={result[mode + "AUROC"][1]:.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            # hover_name=thresholds, # doesn't work as the size is different
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        #fig.update_yaxes(scaleanchor="x", scaleratio=1)
        #fig.update_xaxes(constrain='domain')
        #fig.show()

        # log fig
        artifact_file = "./roc/epoch_" + str(self.state.epoch) + "/roc_curve_epoch_" + str(self.state.epoch) + "_val_" + str(self.state.epoch) + "_" + str(mode) + ".html"
        mlflow.log_figure(fig, artifact_file) 


    def log_prc_graph(self, result, mode = "eval_") -> None:
        precision, recall, thresholds = result[mode + "PrecisionRecallCurve"]# self.val_metric_prc.compute()

        precision = precision.cpu().numpy() if isinstance(precision, torch.Tensor) else precision
        recall = recall.cpu().numpy() if isinstance(recall, torch.Tensor) else recall
        thresholds = thresholds.cpu().numpy() if isinstance(thresholds, torch.Tensor) else thresholds

        fig = px.area(
            x=recall[1], y=precision[1],
#            title=f'Precision-Recall Curve (AUC={self.val_metric_auc.compute():.4f}, AvePrec={self.val_metric_ap.compute():.4f})',
#            title=f'Precision-Recall Curve (AUC={result[mode + "AUC"]:.4f}, AvePrec={result[mode + "AveragePrecision"]:.4f})',
            title=f'Precision-Recall Curve (AvePrec={result[mode + "AveragePrecision"][1]:.4f})',
            labels=dict(x='Recall', y='Precision'),
            # hover_name=thresholds, # doesn't work as the size is different
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        #fig.update_yaxes(scaleanchor="x", scaleratio=1)
        #fig.update_xaxes(constrain='domain')
        #fig.show()

        # log fig
        artifact_file = "./prc/epoch_" + str(self.state.epoch) + "/prc_curve_epoch_" + str(self.state.epoch) + "_val_" + str(self.state.epoch) + "_" + str(mode) + ".html"
        mlflow.log_figure(fig, artifact_file) 
    # TODO: edit and refactor.
    # def _compute_entity_wise_metrics(self, dataloader: DataLoader, prediction_output: EvalLoopOutput) -> PredictionOutput:
    #     """
    #     Compute entity wise metrics.
    #     """
    #     #logger.info("TYPE %s", type(dataloader.dataset))
        
    #     true_predictions, true_labels = align_predictions(prediction_output.predictions, prediction_output.label_ids) #type: ignore

    #     report = classification_report(true_labels, true_predictions)
    #     with open(self.args.output_dir + "/" + 'classification_report_token_wise_' + str(self.state.global_step) + '.txt', 'w') as file:
    #        file.write(report)

    #     predicted_labels = detokenize.preprocess_labels(true_predictions)
    #     valid_labels = detokenize.preprocess_labels(true_labels)
    #     val_tokens = detokenize.preprocess_tokens(dataloader.dataset.return_tokens) #type: ignore
    #     val_offsets = detokenize.extract_offsets(dataloader.dataset.get_features()) #type: ignore

    #     output_lines = detokenize.main(predicted_labels, valid_labels, val_tokens, val_offsets, "test") # TODO: mode
    #     counts = conlleval.evaluate(output_lines)
    #     _result = conlleval.return_result(counts)

    #     file_utils.write_lines(self.args.output_dir, "pred_labels_" + str(self.state.global_step) + ".txt", predicted_labels)
    #     file_utils.write_lines(self.args.output_dir, "valid_labels_" + str(self.state.global_step) + ".txt", valid_labels)
    #     file_utils.write_lines(self.args.output_dir, "val_tokens_" + str(self.state.global_step) + ".txt", val_tokens)
    #     file_utils.write_lines(self.args.output_dir, "val_offsets_" + str(self.state.global_step) + ".txt", val_offsets)
    #     file_utils.write_lines(self.args.output_dir, "output_lines_" + str(self.state.global_step) + ".txt", output_lines)
    #     with open(self.args.output_dir + "/" + 'counts_' + str(self.state.global_step) + '.json', 'w') as json_file:
    #        json_file.write(counts.toJSON())
    #     with open(self.args.output_dir + "/" + 'result_' + str(self.state.global_step) + '.json', 'w') as json_file:
    #        json.dump(_result, json_file)

    #     for class_type in _result["accuracy"]:
    #         prediction_output.metrics[f"eval_entity_wise_accuracy" + class_type] = (_result["accuracy"][class_type]/ 100.0)
    #     for class_type in _result["precision"]:
    #         prediction_output.metrics[f"eval_entity_wise_precision" + class_type] = (_result["precision"][class_type]/ 100.0)
    #     for class_type in _result["recall"]:
    #         prediction_output.metrics[f"eval_entity_wise_recall" + class_type] = (_result["recall"][class_type]/ 100.0)
    #     for class_type in _result["f1"]:
    #         prediction_output.metrics[f"eval_entity_wise_f1" + class_type] = (_result["f1"][class_type]/ 100.0)

    #     return prediction_output

# # TODO: edit and refactor.
# def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
#     """
#     Align predictions.
#     """
    
#     label_map: Dict[int, str] = {i: label for i, label in enumerate(pr.get_labels())}

#     preds = np.argmax(predictions, axis=2)
#     batch_size, seq_len = preds.shape
#     true_labels = [[] for _ in range(batch_size)]
#     predicted_labels = [[] for _ in range(batch_size)]

#     # take only predictions that are not padded.
#     for i in range(batch_size):
#         for j in range(seq_len):
#             if label_ids[i, j] != -100: # TODO: move padding id to somewhere else
#                 true_labels[i].append(label_map[label_ids[i][j]])
#                 predicted_labels[i].append(label_map[preds[i][j]])

#     return predicted_labels, true_labels

def compute_additional_metrics(preds_labels: EvalPrediction) -> Dict:
    """
    In addition to the loss returned in the trainer, also return a dict with
    f1_score, accuracy, precision and recall at evaluation time.

    Args:
      preds_labels (EvalPrediction): Predicted labels

    Returns:
        Dict: Metrics dict with accuracy, balanced accuracy, f1-score, precision, recall
    """
    outputs = np.argmax(preds_labels.predictions, axis=1)
    pred = []
    valid = []

    for pred_label in outputs:
        pred.append(int(pred_label))
    for valid_label in preds_labels.label_ids:
        valid.append(int(valid_label))

    metrics_dict = {}
    metrics_dict['precision'] = precision_score(y_pred=pred, y_true=valid)
    metrics_dict['recall'] = recall_score(y_pred=pred, y_true=valid)
    metrics_dict['f1_score'] = f1_score(y_pred=pred, y_true=valid)
    
    # metrics_dict = {
        # "eval_accuracy": accuracy_score(preds_labels.label_ids, preds_labels.predictions),
        # "eval_precision": precision_score(preds_labels.label_ids, preds_labels.predictions),
        # "eval_recall": recall_score(preds_labels.label_ids, preds_labels.predictions),
        # "eval_f1": f1_score(preds_labels.label_ids, preds_labels.predictions),
    # }
    
    return metrics_dict

class MirnaDiseaseAssocDataset(Dataset):
    """ 
    Helper class to wrap the bc5cdr data into a format that can be easily processed
    by pytorch/the transformers library.
    """

    @staticmethod
    def get_data(file, mode="train"):
        data_file = os.path.join(file)
        instances = pr.get_examples(data_file, mode)
        return MirnaDiseaseAssocDataset(instances)

    @staticmethod
    def get_data_new(file, tokenizer=None, mode="train", with_headers = False, max_seq_length=512, per_device_train_batch_size=32):
        import pandas as pd
        if with_headers:
            data = pd.read_csv(file, sep="\t", header=0, index_col = 0)
        else:
            data = pd.read_csv(file, sep="\t", header=None, names = ["sentence","label"] )
        
        entries = []
        for idx, row in data.iterrows():
            entry = {}
            entry["text_a"] = row["sentence"]
            entry['label'] = str(row["label"]) if mode != "predict" else "0"
            entries.append(entry)

        instances = pr.create_examples(entries, mode)
        dataset = MirnaDiseaseAssocDataset(instances)

        label_list = pr.get_labels() # TODO
        dataset.set_features(label_list, max_seq_length, tokenizer, per_device_train_batch_size, mode)
        return dataset
    
    def __init__(self, instances: List):
        self.instances = instances
        #self.num_labels = len(.label_encoding)
        #self.max_chunks = np.max([len(i.tokens) for i in self.instances])

    def set_features(self, label_list, max_seq_length, tokenizer, batch_size, mode):
        train_features = pr.convert_examples_to_features(self.instances, label_list, max_seq_length, tokenizer, mode)
        
        self.mode = mode
        self.features = train_features
        self.all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.long)
        
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
        item['label_ids'] = tensor(self.features[idx].label_id, dtype=torch.long)
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
class ReModelArguments:
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
class ReDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default=settings.MIRNA_DISEASE_ASSOC_DATA_DIR,
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
class MirnaDisAssocDataTrainingArguments(ReDataTrainingArguments):

    data_dir: str = field(
        default = settings.MIRNA_DISEASE_ASSOC_DATA_DIR,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )

@dataclass
class MultiTaskAssocDataTrainingArguments(ReDataTrainingArguments):
    data_dir: str = field(
        default = settings.MULTI_TASK_ASSOC_DATA_DIR,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )

def re_hyperparameter_space(trial: Trial) -> Dict[str, float]:
    hps =  {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_loguniform('weight_decay', 1e-10, 1e-1),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "seed": trial.suggest_int("seed", 20, 40, 4),
    }
    return hps

def re_compute_objective(metrics: Dict[str, float]) -> float:
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
def create_or_get_optuna_study(o_fold: Optional[int] = None, study_name_prefix = "mirna-disease-assoc", extended_testset: str = None, storage = settings.OTPUNA_STORAGE, direction: str = "maximize") -> Study:
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
