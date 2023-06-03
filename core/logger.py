from pathlib import Path
from typing import Literal

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Logger:
    def __init__(self, name, step_init=0, SUM_FREQ=20):
        self.SUM_FREQ = SUM_FREQ
        self.name = name

        self.total_steps = step_init
        self.epoch = 0
        self.epoch_size = 0

        self.running_loss = {}

        self.writer = None
        self.pbar = None

    def _print_training_status(self):
        for k in self.running_loss:
            self.running_loss[k] /= self.SUM_FREQ

        training_str = f'Ep {self.epoch:3d}'
        if (total_loss := self.running_loss.get('loss', None)) is not None:
            training_str += f'; loss {total_loss:3.3f}'
        self.pbar.set_description(training_str)

        if self.writer is None:
            self.open()

        self.writer.add_scalar('epoch', self.total_steps / self.epoch_size,
                               self.total_steps)
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k],
                                   self.total_steps)
        self.running_loss = {}

    def push(self, metrics):
        self.total_steps += 1
        self.pbar.update(1)

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == 0:
            self._print_training_status()

    def write_dict(self, results, rel: Literal['step', 'epoch'] = 'step'):
        if rel not in ('step', 'epoch'):
            raise ValueError(rel)

        if self.writer is None:
            self.open()

        for key in results:
            self.writer.add_scalar(key, results[key],
                                   self.total_steps if rel == 'step' else self.total_steps / self.epoch_size)

    def open(self):
        rootdir = Path(__file__).parent.parent
        self.writer = SummaryWriter(rootdir.joinpath('runs', self.name))

    def close(self):
        self.closePbar()
        self.writer.close()

    def initPbar(self, epoch_size, epoch, ncols=120):
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.pbar = tqdm(total=epoch_size, desc=f'Ep {epoch:3d}', ncols=ncols)

    def closePbar(self, accuracies=None, lastLR=None):
        if self.pbar is not None:
            if accuracies is not None:
                train, test = accuracies
                desc_str = self.pbar.desc[:-2]
                test_str = f'{test:.3f}'
                if test > 0.87:
                    test_str = '\033[92m' + test_str + '\033[0m'
                desc_str += f', accu ({train:.3f},{test_str}), l-LR {lastLR:.2e}'
                self.pbar.set_description(desc_str)
            self.pbar.close()

    def write_viz(self, image):
        if self.writer is None:
            self.open()

        self.writer.add_image('visualization', image, self.total_steps)
