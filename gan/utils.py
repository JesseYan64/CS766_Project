import os
import json
from datetime import datetime
import torch
from torch import nn

def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    return None

class Logger():
    def __init__(self,
                 exp_name: str='./runs',
                 filename: str=None):
        self.exp_name=exp_name
        self.filename=filename
        self.cache={}
        if not os.path.exists(exp_name):
            os.makedirs(exp_name, exist_ok=True)
        fpath = f"{self.exp_name}/{self.filename}.json"
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                data = f.read()
                print(data)
                self.cache = json.loads(data)
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        
    def add_scalar(self, key: str, value: float, t: int):
        t = str(t)
        if key not in self.cache:
            self.cache[key] = {}
        self.cache[key][t] = value
        self.update()
        return None
    
    def save_weights(self, state_dict, model_name: str='model'):
        fpath = f"{self.exp_name}/{model_name}.pt"
        torch.save(state_dict, fpath)
        return None
    
    def update(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        return None
    
    def close(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        self.cache={}
        return None