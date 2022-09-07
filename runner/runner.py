import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim

from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger

from torch_geometric.loader import DataLoader
from models.DimeNet.model import DimeNet
from models.SphereNet.model import SphereNet
from dataset.dataset_loader import get_dataset


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.logger = get_logger(logger_name=str(config.seed))
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')

        self.train_conf = config.train
        self.batch_size = config.train.batch_size
        self.loss = nn.L1Loss()

        if config.model_name == 'DimeNet':
            self.model = DimeNet(self.config.model)
        elif config.model_name == 'SphereNet':
            self.model = SphereNet(self.config.model)

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

        self._train, self._validation, self._test = get_dataset('./data', mol_state=config.model.mol_state)

    def train(self):
        self.train_dataset = DataLoader(self._train, batch_size=self.batch_size)
        self.valid_dataset = DataLoader(self._validation, batch_size=self.batch_size)

        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        results = defaultdict(list)
        best_val_loss = np.inf

        if self.config.train_resume:
            checkpoint = load_model(self.ck_dir)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_loss = checkpoint['best_valid_loss']
            self.train_conf.epoch -= checkpoint['epoch']

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()

            train_loss = []

            for i, data_batch in enumerate(tqdm(self.train_dataset)):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                out = self.model(z=data_batch.z, pos=data_batch.pos, edge_index=data_batch.edge_index,
                                 batch=data_batch.batch)

                loss = self.loss(out.squeeze(), data_batch.y)
                # backward pass (accumulates gradients).
                loss.backward()

                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(loss.data.cpu().numpy())]

                # display loss
                if (i + 1) % 500 == 0:
                    self.logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, i + 1,
                                                                         float(loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in tqdm(self.valid_dataset):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                with torch.no_grad():
                    out = self.model(z=data_batch.z, pos=data_batch.pos, edge_index=data_batch.edge_index,
                                     batch=data_batch.batch)

                loss = self.loss(out.squeeze(), data_batch.y)
                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()

            results['val_loss'] += [val_loss]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            self.logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            model_snapshot(epoch=epoch, model=self.model, optimizer=optimizer, scheduler=None,
                           best_valid_loss=best_val_loss, exp_dir=self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        submission = pd.read_csv("./data/sample_submission.csv", index_col=0)
        submission = submission.astype('float')

        self.test_dataset = DataLoader(self._test, batch_size=1)

        if self.config.model_name == 'DimeNet':
            self.best_model = DimeNet(self.config.model)
        elif self.config.model_name == 'SphereNet':
            self.best_model = SphereNet(self.config.model)

        best_snapshot = load_model(self.best_model_dir)

        self.best_model.load_state_dict(best_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== test ============================ #
        self.best_model.eval()

        for data_batch in tqdm(self.test_dataset):
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                out = self.model(z=data_batch.z, pos=data_batch.pos, edge_index=data_batch.edge_index,
                                 batch=data_batch.batch)

            if data_batch.state[0] == 'g':
                submission.iloc[int(data_batch.idx[0])]['Reorg_g'] = out.cpu().detach().numpy()
            else:
                submission.iloc[int(data_batch.idx[0])]['Reorg_ex'] = out.cpu().detach().numpy()

        submission.to_csv(self.config.exp_sub_dir + '/submission.csv')
