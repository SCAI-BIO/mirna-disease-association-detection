import os
import torch
from deprecated import deprecated
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
#from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from utils.logger_utils import get_custom_logger

logger = get_custom_logger(__name__)

@deprecated(version='', reason="This method is deprecated, remove references for this method.")
def write_tokens(output_dir, tokens):
    path = os.path.join(output_dir, "token.txt")
    wf = open(path, 'a+')
    for token in tokens:
        if token != "**NULL**":
            wf.write("\n".join(token) + '\n')
    wf.close()

@deprecated(reason="Find better way to save model, such as mlflow.")
def save_model(model, output_dir, tokenizer=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    
    # deprecate the following two lines
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    
    # deprecate the following three lines
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    logger.info('Model was saved to %s', output_dir)

# @deprecated()
# def write_info_file(output_dir, args, info):
#     info_file = os.path.join(output_dir, "info.txt")
#     with open(info_file, 'w') as f:
#         print('Used Training Script:'+'\n'+'train.py'+'\n', file=f)
#         print('Used Pretrained Model: ', file=f)
#         print(args.bert_model+'\n', file=f)
#         print('Used Data: ', file=f)
#         print(args.data_dir+'train.tsv'+'\n', file=f)
#         print('Used Parameters: ', file=f)
#         for key in info.keys():
#             print(key+": "+str(info[key]), file=f)

@deprecated()
def write_labels(output_dir, tags, mode):
    path = os.path.join(output_dir, mode + "_labels.txt")
    f = open(path, 'w')
    for tag in tags:
        for t in tag:
            f.write(t + '\n')
    f.close()

def write_lines(output_dir, output_file, list):
    with open(output_dir + "/" + output_file, 'w') as f:
      for item in list:
        f.write("%s\n" % item)
