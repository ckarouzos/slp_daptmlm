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

from slp.trainer.handlers import CheckpointHandler, EvaluationHandler
from slp.util import from_checkpoint, to_device
from slp.util import log
from slp.util import system

LOGGER = log.getLogger('default')

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
            return y_pred, targets

    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        print_fn = self.pbar.log_message
        self.valid_evaluator.run(dataloader)
        system.print_separator(n=35, print_fn=print_fn)
        metrics = self.valid_evaluator.state.metrics
        print_fn('Target test results')
        system.print_separator(symbol='-', n=35, print_fn=print_fn)
        for name, value in metrics.items():
            print_fn('{:<15} {:<15}'.format(name, value))

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50) -> State:
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

    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x)
        ra.attach(self.trainer, "Train Loss")
        self.pbar.attach(self.trainer, ['Train Loss'])
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED,
                                               self.early_stop)
        ckpt = {
            'model': self.model,
            'optimizer': self.optimizer
        }
        if self.checkpoint_dir is not None:
            self.valid_evaluator.add_event_handler(
                Events.COMPLETED, self.checkpoint, ckpt)

        def graceful_exit(engine, e):
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                LOGGER.warn("CTRL-C caught. Exiting gracefully...")
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

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 50) -> State:
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        #self.val_handler.attach(self.trainer,
        #                        self.valid_evaluator,
        #                        test_loader,
        #                        validation=False,
        #                        test=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

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

class DATrainer(Trainer):
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
        lengths = to_device(batch[3],
                                device=self.device,
                                non_blocking=self.non_blocking)
        return inputs, targets, domains, lengths

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, domains, lengths = self.parse_batch(batch)
        y_pred, d_pred = self.model(inputs, lengths)
        return y_pred, targets, d_pred, domains

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets, d_pred, domains = self.get_predictions_and_targets(batch)
        loss1 = self.loss_fn(y_pred, targets, d_pred, domains, engine.state.epoch)  # type: ignore
        loss_fn2 = da_loss = nn.CrossEntropyLoss()
        loss2 = 0.01 * loss_fn2(d_pred, domains)
        if self.parallel:
            loss1 = loss1.mean()
            loss2 = loss2.mean()
        loss1 = loss1 / self.accumulation_steps
        loss1.backward(retain_graph=self.retain_graph)
        if engine.state.epoch>0:
            loss2 = loss2 / self.accumulation_steps
            loss2.backward(retain_graph=self.retain_graph)
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()  # type: ignore
            self.optimizer.zero_grad()
        loss_value: float = loss1.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets, d_pred, domains = self.get_predictions_and_targets(batch)
            d = {'domain_pred'  : d_pred,
		         'domain_targets'  : domains,
                 'epoch' : engine.state.epoch}
            return y_pred, targets, d

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 50) -> State:
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        #self.val_handler.attach(self.trainer,
        #                        self.valid_evaluator,
        #                        test_loader,
        #                        validation=False,
        #                        test=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

class VADATrainer(Trainer):
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
        lengths = to_device(batch[3],
                                device=self.device,
                                non_blocking=self.non_blocking)
        return inputs, targets, domains, lengths

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, domains, lengths = self.parse_batch(batch)
        y_pred, d_pred = self.model(inputs, lengths)
        return y_pred, targets, d_pred, domains, inputs, lengths

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets, d_pred, domains, inputs, lengths = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets, d_pred, domains, engine.state.epoch, inputs, lengths)  # type: ignore
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
            y_pred, targets, d_pred, domains, inputs, lengths = self.get_predictions_and_targets(batch)
            d = {'domain_pred'  : d_pred,
		        'domain_targets'  : domains,
                'epoch' : engine.state.epoch,
                'inputs' : inputs,
                'lengths' : lengths}
            return y_pred, targets, d
    
    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 50) -> State:
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        #self.val_handler.attach(self.trainer,
        #                        self.valid_evaluator,
        #                        test_loader,
        #                        validation=False,
        #                        test=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)
