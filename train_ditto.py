import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.ditto import train as ditto_train
from ditto_light.ditto_hgat import train as hgat_train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ditto_hgat",
                      choices=["ditto", "ditto_hgat"],
                      help="Choose between original Ditto or enhanced DittoHGAT")
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'

    task = hp.task

    run_tag = '%s_model=%s_lm=%s_size=%s_id=%d' % (
        task, hp.model_type, hp.lm, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    train_dataset = DittoDataset(trainset,
                                lm=hp.lm,
                                max_len=hp.max_len,
                                size=hp.size)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    train_fn = hgat_train if hp.model_type == "ditto_hgat" else ditto_train
    train_fn(trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,
            run_tag=run_tag,
            hp=hp,
            device=device)