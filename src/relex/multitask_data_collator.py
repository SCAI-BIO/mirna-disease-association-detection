import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass, DefaultDataCollator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
import numpy as np
import torch
import transformers
import mlflow
import plotly.express as px
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput, EvalPrediction
from typing import Dict, Optional, List

class NLPDataCollator:
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        first = features[0]
        batch = {}
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DataCollator().collate_batch(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

from torchmetrics import (AUROC, F1Score, ROC, Accuracy, AveragePrecision, 
                          Precision, PrecisionRecallCurve,
                          Recall, ConfusionMatrix, MetricCollection)

class MultitaskTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        metrics = MetricCollection([
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

        self.valid_metrics = {}
        self.valid_metrics["gad"] = metrics.clone(prefix="eval_gad_")
        self.valid_metrics["euadr"] = metrics.clone(prefix="eval_euadr_")
        self.valid_metrics["mirna_disease"] = metrics.clone(prefix="eval_mirna_disease_")
        super().__init__(*args, **kwargs)

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # limit_number_of_samples = 5
        train_sampler = (
#             RandomSampler(train_dataset, num_samples=limit_number_of_samples) # FIXME
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_single_eval_dataloader(self, task_name, eval_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: Evaluation requires a eval_dataset.")

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def get_eval_dataloader(self, eval_dataset = None):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(
            {
                task_name: self.get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.eval_dataset.items()
            }
        )
    
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
        preds_dict = {}
        result = {}
        for task_name in ["euadr", "gad", "mirna_disease"]:
            # print(eval_dataloader.data_loader.collate_fn)
            dataloader_ = dataloader.dataloader_dict[task_name] # .data_loader
            preds_dict[task_name] = super().evaluation_loop(
                dataloader_,
                description=f"Validation: {task_name}",
            )
            logits = preds_dict[task_name].predictions
            labels = preds_dict[task_name].label_ids

            # result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            result[task_name] = self._compute_additional_metrics(dataloader_, preds_dict[task_name], task_name)

        final_result = result["mirna_disease"]
        final_result.metrics["eval_mirna_disease_loss"] = final_result.metrics["eval_loss"]
        final_result.metrics.update(result["gad"].metrics)
        final_result.metrics["eval_gad_loss"] = final_result.metrics["eval_loss"]
        final_result.metrics.update(result["euadr"].metrics) 
        final_result.metrics["eval_euadr_loss"] = final_result.metrics["eval_loss"]
        final_result.metrics.pop("eval_loss", None)

        return final_result

    def _compute_additional_metrics(self, dataloader: DataLoader, prediction_output: EvalLoopOutput, task_name) -> EvalLoopOutput:
        
        assert self.state.epoch is not None
        epoch = int(self.state.epoch) # self.state.global_step
        
        # report = classification_report(prediction_output.label_ids, prediction_output.predictions)
        # mlflow.log_text(report, 'classification_report_token_wise__epoch_' + str(epoch) + '.txt')

        predictions = torch.nn.Softmax(dim=1)(torch.tensor(prediction_output.predictions))
        label_ids = torch.tensor(prediction_output.label_ids)

        self.valid_metrics[task_name].update(predictions, label_ids)
        result = self.valid_metrics[task_name].compute()
        self.valid_metrics[task_name].reset()
        
        assert self.valid_metrics[task_name].prefix is not None
        
        self.log_roc_graph(result, self.valid_metrics[task_name].prefix + "Multiclass")
        self.log_prc_graph(result, self.valid_metrics[task_name].prefix + "Multiclass")
        result.pop(self.valid_metrics[task_name].prefix + 'MulticlassROC', None)
        result.pop(self.valid_metrics[task_name].prefix + 'MulticlassPrecisionRecallCurve', None)
        
        f1_score = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassF1Score', torch.Tensor([-1,-1]))
        result[self.valid_metrics[task_name].prefix + "F1_class_0"] = f1_score[0].item()
        result[self.valid_metrics[task_name].prefix + "F1_class_1"] = f1_score[1].item()
        precision = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassPrecision', torch.Tensor([-1,-1]))
        result[self.valid_metrics[task_name].prefix + "Precision_class_0"] = precision[0].item()
        result[self.valid_metrics[task_name].prefix + "Precision_class_1"] = precision[1].item()
        recall = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassRecall', torch.Tensor([-1,-1]))
        result[self.valid_metrics[task_name].prefix + "Recall_class_0"] = recall[0].item()
        result[self.valid_metrics[task_name].prefix + "Recall_class_1"] = recall[1].item()
        auroc = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassAUROC', torch.Tensor([-1,-1]))
        result[self.valid_metrics[task_name].prefix + "AUROC"] = auroc[0].item()
        average_precision = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassAveragePrecision', torch.Tensor([-1,-1]))
        result[self.valid_metrics[task_name].prefix + "average_precision_class_0"] = average_precision[0].item()
        result[self.valid_metrics[task_name].prefix + "average_precision_class_1"] = average_precision[1].item()
        confmat = result.pop(self.valid_metrics[task_name].prefix + 'MulticlassConfusionMatrix', torch.Tensor([[-1,-1],[-1,-1]]))
        result[self.valid_metrics[task_name].prefix + "ConfMat_TN"] = confmat[0][0].item()
        result[self.valid_metrics[task_name].prefix + "ConfMat_FN"] = confmat[0][1].item()
        result[self.valid_metrics[task_name].prefix + "ConfMat_FP"] = confmat[1][0].item()
        result[self.valid_metrics[task_name].prefix + "ConfMat_TP"] = confmat[1][1].item()


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
