import copy
import logging
import os
import platform
import shutil
import socket
import statistics
import sys
import tempfile
from typing import Any, Dict, List, Optional, Union

import mlflow
import optuna
import transformers
from optuna.trial._frozen import FrozenTrial
from optuna.trial._trial import Trial
from seqeval.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import KFold
from torch.utils.data.dataset import ConcatDataset, Dataset, Subset
# from transformers.modeling_bert import BertForTokenClassification
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer
# from transformers.tokenization_bert_fast import BertTokenizerFast
from transformers.integrations import MLflowCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.megatron_bert.modeling_megatron_bert import \
    MegatronBertForSequenceClassification
from transformers.optimization import (
    get_constant_schedule, get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
from transformers.trainer import Trainer
from transformers.trainer_callback import (TrainerControl, TrainerState, TrainerCallback)
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import BestRun, EvalPrediction, set_seed

import settings
#from transformers.utils.dummy_tokenizers_objects import BertTokenizerFast
from relex.ReTrainer import (
    MirnaDiseaseAssocDataset, ReDataTrainingArguments, ReModelArguments,
    ReTrainer, ReTrainingArguments, compute_additional_metrics,
    create_or_get_optuna_study, re_compute_objective, re_hyperparameter_space)
from utils import logger_utils
from relex import preprocessing as pr
from utils.gpu_usage2 import assign_free_gpus

logger = logger_utils.get_custom_logger(__name__, logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

if platform.system() == "Linux" and (socket.gethostname().startswith("aiolos")):
    assign_free_gpus(threshold_vram_usage=1500, max_gpus=1, wait=False, sleep_time=10)

n_total_inner_folds = 5
n_total_optuna_trials = 120

distributions = {
    "gradient_accumulation_steps": optuna.distributions.IntDistribution(1, 8, step=1),
    # "per_device_train_batch_size": optuna.distributions.CategoricalDistribution([4 , 8, 16, 32, 64]),
    "hidden_dropout_prob": optuna.distributions.FloatDistribution(0.1, 0.6, step=0.1),
    "learning_rate": optuna.distributions.FloatDistribution(1e-5, 5e-5, log=True),
    "weight_decay": optuna.distributions.FloatDistribution(1e-10, 1e-1, log=True),
    "adam_epsilon": optuna.distributions.FloatDistribution(1e-10, 1e-6, log=True),
    "seed": optuna.distributions.IntDistribution(10, 42, step=4),
}

class MetricsCallback(TrainerCallback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.epoch_metrics = []
        self.epochs = []
        self.mlflow_run_id = ""
        self.mlflow_experiment_id = ""

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.pop("metrics")
        self.epochs.append(copy.deepcopy(state.epoch))
        self.epoch_metrics.append(copy.deepcopy(metrics))
        run = mlflow.active_run()
        assert run is not None
        self.mlflow_run_id = run.info.run_id
        self.mlflow_experiment_id = run.info.experiment_id
        logger.info("Training started, check out run: %s", settings.MLFLOW_TRACKING_URI + "/#/experiments/" + run.info.experiment_id + "/runs/" + run.info.run_id)

        # print("Active run_id: {}".format(run.info.run_id))
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        #run = mlflow.active_run()
        mlflow.log_text(str(kwargs.pop("model")), "./model/model_summary.txt")
        return super().on_train_begin(args, state, control, **kwargs)

def train_model(re_training_args, tokenizer, train_dataset, eval_dataset, trial) -> Dict[str, Any]:
    logger.info(f"Setting trial.params {trial.params}")
    for k,v in trial.params.items():
        setattr(re_training_args, k, v)
        
    re_training_args.run_name = re_training_args.experiment_name + "-trial-" + str(trial.number)
    re_training_args.optuna_trial_id = trial._trial_id
    re_training_args.optuna_trial_number = trial.number
    re_training_args.optuna_study_name = trial.study.study_name
    re_training_args.optuna_study_id = trial._study_id
    re_training_args.output_dir = tempfile.mkdtemp()
    logger.info("Setting output dir to %s", re_training_args.output_dir)

    logger.info("Start mlflow tracking.")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(re_training_args.experiment_name)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = re_training_args.experiment_name
    
    logger.info("Build trainer.")
    from relex.multitask_model import BertForSequenceClassification, MegatronBertForSequenceClassificationModified
    from relex.preprocess import convert_to_features
    from relex.multitask_data_collator import MultitaskTrainer, NLPDataCollator
    multitask_model = MegatronBertForSequenceClassificationModified.from_pretrained(
        "EMBO/BioMegatron345mCased",
        task_labels_map={"gad": 2, "euadr": 2, "mirna_disease": 2},
    )
    
    train_args = transformers.TrainingArguments(output_dir=re_training_args.output_dir)
    for arg in vars(re_training_args):
        setattr(train_args, arg, getattr(re_training_args, arg))
    
    train_args.overwrite_output_dir=True
    train_args.do_train=True
    train_args.save_steps=3000
    train_args.label_names = ["label_ids"]
    train_args.evaluation_strategy="epoch"
    
    trainer = MultitaskTrainer(
        model=multitask_model,
        # args=transformers.TrainingArguments(
        #     output_dir=args.output_dir,
        #     run_name = re_training_args.run_name,
        #     overwrite_output_dir=True,
        #     learning_rate=1e-5,
        #     do_train=True,
        #     num_train_epochs=args.num_train_epochs,
        #     # Adjust batch size if this doesn't fit on the Colab GPU
        #     per_device_train_batch_size=args.per_device_train_batch_size,
        #     save_steps=3000,
        #     label_names = ["label_ids"],
        #     evaluation_strategy="epoch"
        # ),
        args = train_args,
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks = [MetricsCallback],
    )
    trainer.train()
    
    val_losses = []
    val_auroc = []
    epochs = []
    
    metrics_callback: Optional[MetricsCallback] = trainer.pop_callback(MetricsCallback)
    
    assert metrics_callback is not None
    
    for epoch, metrics in zip(metrics_callback.epochs, metrics_callback.epoch_metrics):
        epochs.append(epoch)
        val_losses.append(metrics["eval_mirna_disease_loss"])
        val_auroc.append(metrics["eval_mirna_disease_AUROC"])

    #  calculate best result based on min. val_loss, not the last one.
    index_max = max(range(len(val_auroc)), key=val_auroc.__getitem__)
    
    result = dict()
    result["val_auroc"] = val_auroc[index_max]
    result["val_loss"] = val_losses[index_max]
    result["best_epoch"] = epochs[index_max]
    result["mlflow_run_id"] = metrics_callback.mlflow_run_id
    result["mlflow_experiment_id"] = metrics_callback.mlflow_experiment_id
    
    return result

if __name__ == "__main__":
    logger.info("Working on host %s", socket.gethostname())
    
    from relex.arguments import parse_args
    args = parse_args()

    re_training_args = args
    
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger_utils.reset_transformers_logger(logger)
    logger_utils.reset_optuna_logger(logger)

    set_seed(re_training_args.seed) # call also here
    
    inner_study = create_or_get_optuna_study(study_name_prefix=re_training_args.experiment_name)
    
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EMBO/BioMegatron345mCased")
    dataset_dict = {
        "gad": {
            "train": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/raw/GAD/1/train.tsv", mode="train", max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
            "validation": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/raw/GAD/1/test.tsv", mode="test", with_headers=True, max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
        },
        "euadr": {
            "train": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/raw/euadr/1/train.tsv", mode="train", max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
            "validation": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/raw/euadr/1/test.tsv", mode="test", with_headers=True, max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
        },
        "mirna_disease": {
                "train": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/miRNA_disease_relations/train.tsv", mode="train", max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
                "validation": MirnaDiseaseAssocDataset.get_data_new(settings.BASE_DATA_DIR  +"/RE/miRNA_disease_relations/test.tsv", mode="test", max_seq_length=512, per_device_train_batch_size=args.per_device_train_batch_size, tokenizer=tokenizer),
        },
    }

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in dataset_dict.items()
    }
    eval_dataset = {
        task_name: dataset["validation"] for task_name, dataset in dataset_dict.items()
    }
    
    for i_trial in range(0, n_total_optuna_trials):
        # Check: stop if all optuna trials are completed
        all_completed_trials = inner_study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if len(all_completed_trials) == n_total_optuna_trials:
            logger.warn("Optuna loop total trials completed.")
            continue
        
        # Check: only execute a certain trial
        if i_trial != args.i_trial and args.i_trial != -1:
            continue
        
        trial: Trial = inner_study.ask(distributions)
        logger.info("Trial started: %s with id %s", i_trial, trial._trial_id)

        results = train_model(re_training_args, tokenizer, train_dataset, eval_dataset, trial)
        
        trial_results = dict()
        # if "val_f1s" in trial.user_attrs.keys():
        # trial_results["val_f1s"] = trial_user_attributes["val_f1s", []) + [results["val_f1"]]
        # trial_results["val_losses"] = trial_user_attributes["val_losses", []) + [results["val_loss"]]
        # trial_results["mlflow_run_ids"] = trial_user_attributes["mlflow_run_ids", []) + [results["mlflow_run_id"]]
        # trial_results["mlflow_experiment_ids"] = trial_user_attributes["mlflow_experiment_ids", []) + [results["mlflow_experiment_id"]]
        # trial_results["best_epochs"] = trial_user_attributes["best_epochs", []) + [results["best_epoch"]]
        # trial_results["i_i_fold"] = trial_user_attributes["i_i_fold", []) + [i_i_fold]

        logger.info('Finishing (for trial %s) with an auroc of %s and val_loss of %s at best epoch %s', 
                    i_trial, results["val_auroc"], results["val_loss"], results["best_epoch"])
        
        trial.set_user_attr("val_auroc",results["val_auroc"])
        trial.set_user_attr("val_loss",results["val_loss"])
        trial.set_user_attr("mlflow_run_id",results["mlflow_run_id"])
        trial.set_user_attr("mlflow_experiment_id",results["mlflow_experiment_id"])
        trial.set_user_attr("best_epoch",results["best_epoch"])
        
        inner_study.tell(trial, results["val_auroc"])
        logger.info("Optuna trial (with number %s) result (%s).", str(trial.number), str(results["val_auroc"]))
        logger.info("Trial finished: %s", i_trial)

    logger.info("Reporting for the best trial (number: %s, study: %s) the value : %s", str(inner_study.best_trial.number), str(inner_study.study_name), str(inner_study.best_trial.value))
    logger.info("Reporting for the best trial (number: %s, study: %s) the params: %s", str(inner_study.best_trial.number), str(inner_study.study_name), str(inner_study.best_trial.params))
