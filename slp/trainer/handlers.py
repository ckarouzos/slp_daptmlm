import shutil

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from slp.util import system
from slp.util import types


class CheckpointHandler(ModelCheckpoint):
    """Augment ignite ModelCheckpoint Handler with copying the best file to a
    {filename_prefix}_{experiment_name}.best.pth.
    This helps for automatic testing etc.
    Args:
        engine (ignite.engine.Engine): The trainer engine
        to_save (dict): The objects to save
    """
    def __call__(self, engine: Engine, to_save: types.GenericDict) -> None:
        super(CheckpointHandler, self).__call__(engine, to_save)
        # Select model with best loss
        _, paths = self._saved[-1]
        for src in paths:
            splitted = src.split('_')
            fname_prefix = splitted[0]
            name = splitted[1]
            dst = f'{fname_prefix}_{name}.best.pth'
            shutil.copy(src, dst)


class EvaluationHandler(object):
    def __init__(self, pbar: Optional[ProgressBar] = None,
                 validate_every: int = 1,
                 early_stopping: Optional[EarlyStopping] = None):
        self.validate_every = validate_every
        self.print_fn = pbar.log_message if pbar is not None else print
        self.early_stopping = early_stopping

    def __call__(self, engine: Engine, evaluator: Engine,
                 dataloader: DataLoader, validation: bool = True):
        if engine.state.epoch % self.validate_every != 0:
            return
        evaluator.run(dataloader)
        system.print_separator(n=35, print_fn=self.print_fn)
        metrics = evaluator.state.metrics
        phase = 'Validation' if validation else 'Training'
        self.print_fn('Epoch {} {} results'
                      .format(engine.state.epoch, phase))
        system.print_separator(symbol='-', n=35, print_fn=self.print_fn)
        for name, value in metrics.items():
            self.print_fn('{:<15} {:<15}'.format(name, value))

        if validation and self.early_stopping:
            loss = self.early_stopping.best_score
            patience = self.early_stopping.patience
            cntr = self.early_stopping.counter
            self.print_fn('{:<15} {:<15}'.format('best loss', -loss))
            self.print_fn('{:<15} {:<15}'.format('patience left',
                                                 patience - cntr))
            system.print_separator(n=35, print_fn=self.print_fn)

    def attach(self, trainer: Engine, evaluator: Engine,
               dataloader: DataLoader, validation: bool = True):
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            self, evaluator, dataloader,
            validation=validation)

class PeriodicNewbob(object):
    def __init__(self, period):
        self.period = period

    def __call__(self, engine: Engine, optimizer):
        if engine.state.epoch % self.period == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2.

    def attach(self, engine: Engine, optimizer):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self, optimizer)

class AugmentDataset(object):

    def predictions(self, model, unlabeled_loader, device):
        model.eval()
        pos = []
        neg = []
        soft = nn.Softmax(dim=1)
        with torch.no_grad():
            for index, batch in enumerate(unlabeled_loader):
                inp = batch[0][0]
                inputs = batch[0].to(device)
                pred = soft(model(inputs)[0])
                if pred[0][1] > pred[0][0]:
                    pos.append((inp, pred[0][1]))
                else:
                    neg.append((inp, pred[0][0]))
        pos.sort(reverse=True, key=lambda tup: tup[1])
        neg.sort(reverse=True, key=lambda tup: tup[1])
        top = 0.5
        toppos = int(np.floor(top * len(pos)))
        topneg = int(np.floor(top * len(neg)))
        spos = pos[:toppos]
        sneg = neg[:topneg]
        selected = []
        labels = []
        for a,b in sneg:
            selected.append(a)
            labels.append(0)
        for a,b in spos:
            selected.append(a)
            labels.append(1)
        npos = pos[toppos:]
        nneg = neg[topneg:]
        unlabeled = []
        for a,b in nneg:
            unlabeled.append(a)
        for a,b in npos:
            unlabeled.append(a)
        return selected, labels, unlabeled

    def __call__(self, engine: Engine, dataloader: DataLoader, unlabeled_loader: DataLoader, model, device):
        #predict new labels
        well_predicted, new_labels, unlabaled = self.predictions(model, unlabeled_loader, device)
        #add to labeled dataset + sampler
        old_len = len(dataloader.dataset)
        dataloader.dataset.reviews = dataloader.dataset.reviews + well_predicted
        dataloader.dataset.labels = dataloader.dataset.labels + new_labels
        new_len = len(dataloader.dataset)
        new_indices = list(range(old_len, new_len))

        dataloader.sampler.indices = torch.cat((dataloader.sampler.indices, torch.tensor(new_indices)))
        #create new unlabeled
        unlabeled_loader.dataset.reviews = unlabaled
        unlabeled_loader.dataset.labels = [-1] * len(unlabaled)
        new_len  = len(unlabeled_loader.dataset)
        new_indices = list(range(new_len))
        unlabeled_loader.sampler.indices = torch.tensor(new_indices)

    def attach(self, engine: Engine, dataloader: DataLoader, unlabeled_loader: DataLoader, model, device):
        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self, dataloader, unlabeled_loader, 
            model, device)
