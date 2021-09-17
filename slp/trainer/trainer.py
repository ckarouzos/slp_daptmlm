import os
from typing import Union
import torch
import torch.nn as nn

from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, State
from ignite.metrics import RunningAverage, Loss

from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from typing import cast, List, Optional, Tuple, TypeVar
from slp.util import types
from slp.util.parallel import DataParallelModel, DataParallelCriterion

from slp.trainer.handlers import CheckpointHandler, EvaluationHandler, PeriodicNewbob, AugmentDataset
from slp.util import from_checkpoint, to_device
from slp.util import log
from slp.util import system


TrainerType = TypeVar('TrainerType', bound='Trainer')


class Trainer(object):
    def __init__(self: TrainerType,
                 model: nn.Module,
                 optimizer: Optimizer,
                 checkpoint_dir: str = '../../checkpoints',
                 experiment_name: str = 'experiment',
                 model_checkpoint: Optional[str] = None,
                 optimizer_checkpoint: Optional[str] = None,
                 metrics: types.GenericDict = None,
                 patience: int = 10,
                 validate_every: int = 1,
                 accumulation_steps: int = 1,
                 loss_fn: Union[_Loss, DataParallelCriterion] = None,
                 non_blocking: bool = True,
                 retain_graph: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: str = 'cpu',
                 parallel: bool = False) -> None:
        self.dtype = dtype
        self.retain_graph = retain_graph
        self.non_blocking = non_blocking
        self.device = device
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        model_checkpoint = self._check_checkpoint(model_checkpoint)
        optimizer_checkpoint = self._check_checkpoint(optimizer_checkpoint)

        self.model = cast(nn.Module, from_checkpoint(
                model_checkpoint, model, map_location=torch.device('cpu')))
        self.model = self.model.type(dtype).to(device)
        self.optimizer = from_checkpoint(optimizer_checkpoint, optimizer)
        self.parallel = parallel
        if parallel:
            if device == 'cpu':
                raise ValueError("parallel can be used only with cuda device")
            self.model = DataParallelModel(self.model).to(device)
            self.loss_fn = DataParallelCriterion(self.loss_fn)  # type: ignore
        if metrics is None:
            metrics = {}
        if 'loss' not in metrics:
            if self.parallel:
                metrics['loss'] = Loss(
                    lambda x, y: self.loss_fn(x, y).mean())  # type: ignore
            else:
                metrics['loss'] = Loss(self.loss_fn)
        self.trainer = Engine(self.train_step)
        self.train_evaluator = Engine(self.eval_step)
        self.valid_evaluator = Engine(self.eval_step)
        for name, metric in metrics.items():
            metric.attach(self.train_evaluator, name)
            metric.attach(self.valid_evaluator, name)

        self.pbar = ProgressBar()
        self.val_pbar = ProgressBar(desc='Validation')

        if checkpoint_dir is not None:
            self.checkpoint = CheckpointHandler(
                checkpoint_dir, experiment_name, score_name='validation_loss',
                score_function=self._score_fn, n_saved=2,
                require_empty=False, save_as_state_dict=True)

        self.early_stop = EarlyStopping(
            patience, self._score_fn, self.trainer)

        self.val_handler = EvaluationHandler(pbar=self.pbar,
                                             validate_every=1,
                                             early_stopping=self.early_stop)
        self.attach()
        log.info(
            f'Trainer configured to run {experiment_name}\n'
            f'\tpretrained model: {model_checkpoint} {optimizer_checkpoint}\n'
            f'\tcheckpoint directory: {checkpoint_dir}\n'
            f'\tpatience: {patience}\n'
            f'\taccumulation steps: {accumulation_steps}\n'
            f'\tnon blocking: {non_blocking}\n'
            f'\tretain graph: {retain_graph}\n'
            f'\tdevice: {device}\n'
            f'\tmodel dtype: {dtype}\n'
            f'\tparallel: {parallel}')

    def _check_checkpoint(self: TrainerType,
                          ckpt: Optional[str]) -> Optional[str]:
        if ckpt is None:
            return ckpt
        if system.is_url(ckpt):
            ckpt = system.download_url(cast(str, ckpt), self.checkpoint_dir)
        ckpt = os.path.join(self.checkpoint_dir, ckpt)
        return ckpt

    @staticmethod
    def _score_fn(engine: Engine) -> float:
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates
            the val loss

        Returns:
            (float): The validation loss
        """
        negloss: float = -engine.state.metrics['loss']
        return negloss

    def parse_batch(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        y_pred = self.model(inputs)
        return y_pred, targets

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)  # type: ignore
        if self.parallel:
            loss = loss.mean()
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets = self.get_predictions_and_targets(batch)
            #import ipdb; ipdb.set_trace()
            if self.parallel:
                y_pred = torch.reshape(torch.stack(tuple(y_pred)), (targets.size(0),2))
            return y_pred, targets

    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        return self.valid_evaluator.run(dataloader)

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50) -> State:
        log.info(
            'Trainer will run for\n'
            f'model: {self.model}\n'
            f'optimizer: {self.optimizer}\n'
            f'loss: {self.loss_fn}')
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

    def overfit_single_batch(self: TrainerType,
                             train_loader: DataLoader) -> State:
        single_batch = [next(iter(train_loader))]

        if self.trainer.has_event_handler(self.val_handler, Events.EPOCH_COMPLETED):
            self.trainer.remove_event_handler(self.val_handler, Events.EPOCH_COMPLETED)

        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                single_batch,  # type: ignore
                                validation=False)
        out = self.trainer.run(single_batch, max_epochs=100)
        return out

    def fit_debug(self: TrainerType,
                  train_loader: DataLoader,
                  val_loader: DataLoader) -> State:
        train_loader = iter(train_loader)
        train_subset = [next(train_loader), next(train_loader)]
        val_loader = iter(val_loader)  # type: ignore
        val_subset = [next(val_loader), next(val_loader)]  # type ignore
        out = self.fit(train_subset, val_subset, epochs=6)  # type: ignore
        return out

    def _attach_checkpoint(self: TrainerType) -> TrainerType:
        ckpt = {
            'model': self.model,
            'optimizer': self.optimizer
        }
        if self.checkpoint_dir is not None:
            self.valid_evaluator.add_event_handler(
                Events.COMPLETED, self.checkpoint, ckpt)
        return self


    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x)
        ra.attach(self.trainer, "Train Loss")
        self.pbar.attach(self.trainer, ['Train Loss'])
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED,
                                               self.early_stop)
        self = self._attach_checkpoint()
        def graceful_exit(engine, e):
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                log.warn("CTRL-C caught. Exiting gracefully...")
            else:
                raise(e)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.train_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        self.valid_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        return self


class AutoencoderTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, inputs


class SequentialTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        lengths = to_device(batch[2],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets, lengths

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, lengths = self.parse_batch(batch)
        y_pred = self.model(inputs, lengths)
        return y_pred, targets


class Seq2seqTrainer(SequentialTrainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, inputs, lengths


class TransformerTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        mask_inputs = to_device(batch[2],
                                device=self.device,
                                non_blocking=self.non_blocking)
        mask_targets = to_device(batch[3],
                                 device=self.device,
                                 non_blocking=self.non_blocking)
        return inputs, targets, mask_inputs, mask_targets

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, mask_inputs, mask_targets = self.parse_batch(batch)
        y_pred = self.model(inputs,
                            targets,
                            source_mask=mask_inputs,
                            target_mask=mask_targets)
        targets = targets.view(-1)
        y_pred = y_pred.view(targets.size(0), -1)
        # TODO: BEAMSEARCH!!
        return y_pred, targets

class BertTrainer(Trainer):
    def __init__(self, *args, newbob_period=1, **kwargs):
        super(BertTrainer, self).__init__(*args, **kwargs)
        self.newbob = PeriodicNewbob(newbob_period)
        self.newbob.attach(self.valid_evaluator, self.optimizer)

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                            device=self.device,
                            non_blocking=self.non_blocking)
        target = to_device(batch[1],
                           device=self.device,
                           non_blocking=self.non_blocking)
        segms = to_device(batch[2],
                           device=self.device,
                           non_blocking=self.non_blocking)
        attention_masks = to_device(batch[3],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, target, segms, attention_masks

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, target, _, _ = self.parse_batch(batch)
        #import ipdb; ipdb.set_trace()
        #output = [x[0] for x in self.model(inputs)]
        output = self.model(inputs)[0]
        #if not self.model.training:
        #    import ipdb; ipdb.set_trace()
        #output = torch.squeeze(output, dim=1)
        return output, target

class DoubleBertTrainer(Trainer):
    def __init__(self, *args, newbob_period=1, **kwargs):
        super(DoubleBertTrainer, self).__init__(*args, **kwargs)
        self.newbob = PeriodicNewbob(newbob_period)
        self.newbob.attach(self.valid_evaluator, self.optimizer)
    
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                            device=self.device,
                            non_blocking=self.non_blocking)
        target = to_device(batch[1],
                           device=self.device,
                           non_blocking=self.non_blocking)
        domain = to_device(batch[2],
                            device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, target, domain
    
    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, target, domains = self.parse_batch(batch)
        #import ipdb; ipdb.set_trace()
        loss, output = self.model(inputs, source=domains[0], labels=target)
        return loss, output, target, domains

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        loss, y_pred, targets, domains = self.get_predictions_and_targets(batch)
        #loss = self.loss_fn(y_pred, targets, domains)  # type: ignore
        #if self.parallel:
        #    loss = loss.mean()
        #import ipdb; ipdb.set_trace()
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 2) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            loss, y_pred, targets, domains = self.get_predictions_and_targets(batch)
            d = {'domains'  : domains}
            return y_pred, targets, d


class AugmentBertTrainer(Trainer):
    def __init__(self, *args, newbob_period=1, **kwargs):
        super(AugmentBertTrainer, self).__init__(*args, **kwargs)
        self.newbob = PeriodicNewbob(newbob_period)
        self.newbob.attach(self.valid_evaluator, self.optimizer)
        self.augment = AugmentDataset()

    def fit(self: TrainerType,
        train_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50) -> State:
        log.info(
            'Trainer will run for\n'
            f'model: {self.model}\n'
            f'optimizer: {self.optimizer}\n'
            f'loss: {self.loss_fn}')
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        self.augment.attach(self.trainer, train_loader, unlabeled_loader, self.model, self.device)

        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                            device=self.device,
                            non_blocking=self.non_blocking)
        target = to_device(batch[1],
                           device=self.device,
                           non_blocking=self.non_blocking)
        segms = to_device(batch[2],
                           device=self.device,
                           non_blocking=self.non_blocking)
        attention_masks = to_device(batch[3],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, target, segms, attention_masks

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, target, _, _ = self.parse_batch(batch)
        output = self.model(inputs)[0]
        return output, target

class BertVADATrainer(Trainer):
    def __init__(self, *args, newbob_period=1, **kwargs):
        super(BertVADATrainer, self).__init__(*args, **kwargs)
        self.newbob = PeriodicNewbob(newbob_period)
        self.newbob.attach(self.valid_evaluator, self.optimizer)
    
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                            device=self.device,
                            non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                           device=self.device,
                           non_blocking=self.non_blocking)
        domains = to_device(batch[2],
                            device=self.device,
                            non_blocking=self.non_blocking)
        segms = to_device(batch[3],
                           device=self.device,
                           non_blocking=self.non_blocking)
        attention_masks = to_device(batch[4],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, targets, domains, segms, attention_masks

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, domains, segms, attention_masks  = self.parse_batch(batch)
        y_pred, d_pred = self.model(inputs)
        return y_pred, targets, d_pred, domains, inputs

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets, d_pred, domains, inputs = self.get_predictions_and_targets(batch)
        #import ipdb; ipdb.set_trace()
        loss = self.loss_fn(y_pred, targets, d_pred, domains, inputs, engine.state.epoch, self.trainer.state.iteration)  # type: ignore
        if self.parallel:
            loss = loss.mean()
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets, d_pred, domains, inputs = self.get_predictions_and_targets(batch)
            #import ipdb; ipdb.set_trace()
            d = {'domain_pred'  : d_pred,
		        'domain_targets'  : domains,
                'inputs' : inputs,
                'epoch' : engine.state.epoch}
            return y_pred, targets, d
      
