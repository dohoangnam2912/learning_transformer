from pathlib import Path

"""
Define 2 methods: Get config and map the path to save the weights of the model
"""

def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350, #Eng to Italian is 350, check for another pairs you would like to train
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None, #Restart training later if it crashes
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel" #Experiment name for tensorboard on which we will save the losses
    }

def get_weight_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)
