import warnings
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from . import util
from . import sinkhornknopp as sk

warnings.simplefilter("ignore", UserWarning)


class ModelCluster:
    def __init__(self, model, hc, cluster_num, train_loader, num_epochs, lr, lrdrop, weight_decay=1e-5, batch_size=64, use_cpu=False):
        self.model = model
        self.hc = hc
        self.cluster_num = cluster_num
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.use_cpu = use_cpu
        self.lr_schedule = lambda epoch: ((epoch < 350) * (self.lr * (0.1 ** (epoch // lrdrop))) +
                                          (epoch >= 350) * self.lr * 0.1 ** 3)
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        self.model.to(self.device)
        self.L = None
        self.optimize_times = []

    def optimize_labels(self, niter):
        sk.gpu_sk(self)
        self.PS = 0

    def predict(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.num_epochs):
            self.optimize_epoch(optimizer, self.train_loader, epoch, validation=False)

        predict_labels, features = self._get_predict_labels_and_features()
        centers = self._calculate_centers(features, predict_labels)
        return predict_labels, centers

    def _get_predict_labels_and_features(self):
        self.model.eval()
        all_features = []
        all_outputs = []

        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                all_outputs.append(outputs)
                all_features.append(data)

        all_outputs = torch.cat(all_outputs)
        all_features = torch.cat(all_features)
        predict_labels = torch.argmax(all_outputs, dim=1).cpu().numpy()
        return predict_labels, all_features

    def _calculate_centers(self, features, predict_labels):
        centers = []
        for i in range(self.cluster_num):
            indices = np.where(predict_labels == i)[0]
            cluster_features = features[indices]
            center = cluster_features.mean(dim=0)
            centers.append(center.cpu().numpy())

        return np.array(centers)

    def optimize_epoch(self, optimizer, loader, epoch, validation=False):
        loss_value = util.AverageMeter()
        self.model.train()
        lr = self.lr_schedule(epoch)
        XE = torch.nn.CrossEntropyLoss()
        for iter, (data, label) in enumerate(loader):
            niter = epoch * len(loader) + iter
            if self.optimize_times and niter * self.batch_size >= self.optimize_times[-1]:
                self.model.headcount = 1
                print('Optimization starting', flush=True)
                with torch.no_grad():
                    _ = self.optimize_times.pop()
                    self.optimize_labels(niter)
            data = data.to(self.device)
            label = label.to(self.device)
            mass = data.size(0)
            optimizer.zero_grad()
            final = self.model(data)

            if self.hc == 1:
                # print(type(final))
                # print(type(label))
                # print(final.device)
                # print(label.device)

                # print(final.dtype)
                # print(label.dtype)           
                
                #print('final, ',final.shape)
                
                # final = final.to(torch.int64)
                label = label.to(torch.int64)
                loss = XE(final, label)
            else:
                assert 0, 'unexpected else'
                loss = torch.mean(torch.stack([XE(final[h], label) for h in range(self.hc)]))




            loss.backward()
            optimizer.step()
            loss_value.update(loss.item(), mass)

        return {'loss': loss_value.avg}
