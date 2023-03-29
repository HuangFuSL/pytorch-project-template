import time

import torch
import torch.backends.cudnn
import torch.cuda


def saveModelDict(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)


def loadModelDict(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))


def setSeed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resetSeed():
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed(int(time.time()))
    torch.cuda.manual_seed_all(int(time.time()))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
