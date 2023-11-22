
from functools import reduce

import warnings


import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from . import util
from . import models
from .data1 import get_standard_data_loader_pairs,get_standard_data_loader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.simplefilter("ignore", UserWarning)


class Probes(nn.Module):
    """Linear probing container."""
    def __init__(self, trunk, probed_layers, num_classes, embd_size):
        
        # print('probed_layers: ',probed_layers)
        super(Probes, self).__init__()
        self.trunk = trunk
        self.probed_layers = probed_layers
        self.probes = nn.ModuleList()
        self.embd_size = embd_size
        x = torch.zeros(444,embd_size)
        num_classes = num_classes
        
        self.adapt_layer = nn.Linear(embd_size, 256 * 6 * 6)
        
        
        
        n_lin = 9200
        cnvs = [nn.MaxPool2d(6, stride=6, padding=3),
                nn.MaxPool2d(4, stride=4, padding=0),
                nn.MaxPool2d(3, stride=3, padding=1),
                nn.MaxPool2d(3, stride=3, padding=1),
                nn.MaxPool2d(2, stride=2, padding=0)]
        # sizess = [9600, 9216, 9600, 9600,9216]

        self.deepest_layer_index = 0
        j=0
        layer_list = self.trunk.modules()
        for index, (name) in enumerate(list(layer_list)[0].children()):  # named_children
            # print(f'index: {index}, name: {name}')
            x = name.forward(x)
            # print(f'Visiting layer {index: 3d}: {name} [shape: {x.shape}] ()')
            if index > max(self.probed_layers):
                break
            if index in self.probed_layers or name in self.probed_layers:
                self.deepest_layer_index = index
                # Downsampler
                # x_volume = reduce(lambda x, y: x * y, x.shape[1:])
                # downsampler = cnvs[j] #
                # j+=1
                # y = downsampler(x)
                # y_volume = reduce(lambda x, y: x * y, y.shape[1:])

                # # Linear classifier
                # bn = nn.BatchNorm2d(y.shape[1], affine=False)
                # predictor = nn.Conv2d(y.shape[1], num_classes, y.shape[2:4], bias=True)
                # torch.nn.init.xavier_uniform_(predictor.weight, gain=1)
                # torch.nn.init.constant_(predictor.bias, 0)
                # Probe
                # self.probes.append(nn.Sequential(downsampler, bn, predictor))
                
                print(x.shape)
                self.probes.append(nn.Linear(x.shape[1],x.shape[1]))
                

    def forward(self, x):
        # x = self.adapt_layer(x)

        outputs = []
        for index, (name, layer) in enumerate(self.trunk.named_children()):
            # print(f'index: {index}, name: {name}, layer: {layer}')
            x = layer.forward(x)
            probe_index = None
            if index in self.probed_layers:
                probe_index = self.probed_layers.index(index)
                # print(probe_index)
            elif name in self.probed_layers:
                probe_index = self.probed_layers.index(name)
            if probe_index is not None:
                # print('x_shape: ',x.shape)
                y = self.probes[probe_index](x).squeeze()
                # outputs += [y]
                outputs = y
            if index == self.deepest_layer_index:
                break
        return outputs

    def lp_parameters(self):
        return self.probes.parameters()


def model_with_probes(model_path, which='Imagenet', arch='alexnet'):
    if which == 'Imagenet':
        nc = 1000
    elif which == 'Places':
        nc = 205
    state_dict = torch.load(model_path) # ['state_dict']
    ncls = []
    for q in (state_dict.keys()):
        if 'top_layer' in q:
            if 'weight' in q:
                ncl = state_dict[q].shape[0]
                ncls.append(ncl)
    outs = ncls
    model = models.__dict__[arch](num_classes=outs)
    model.load_state_dict(state_dict)
    layers = [1, 4, 7, 9, 11]  # because BN.
    util.search_absorb_bn(model)
    model = util.sequential_skipping_bn_cut(model)
    for relu in filter(lambda x: issubclass(x.__class__, nn.ReLU), model.children()):
        relu.inplace = False
    model = Probes(model, layers, num_classes=nc)
    return model


class LinearProbesOptimizer():
    def __init__(self):
        self.num_epochs = 36
        self.lr = 0.01
        def zheng_lr_schedule(epoch):
            if epoch < 10:
                return 1e-2
            elif epoch < 20:
                return 1e-3
            elif epoch < 30:
                return 1e-4
            else:
                return 1e-5
        self.lr_schedule = lambda epoch: zheng_lr_schedule(epoch)
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.nesterov = False
        self.validate_only = False

    def optimize(self, model, train_loader, val_loader=None, optimizer=None):
        criterion = self.criterion
        model = model.to('cuda:0')

        if optimizer is None:
            optimizer = self.get_optimizer(model)

        # Perform epochs
        for epoch in range(self.num_epochs):
            self.optimize_epoch(model, criterion, optimizer, train_loader, epoch, is_validation=False)
            if epoch > 25 and val_loader:
                with torch.no_grad():
                    self.optimize_epoch(model, criterion, optimizer, val_loader, epoch, is_validation=True)

        return model

    def get_optimizer(self, model):
        return torch.optim.SGD(model.lp_parameters(),
                               lr=self.lr_schedule(0),
                               momentum=self.momentum,
                               weight_decay=self.weight_decay,
                               nesterov=self.nesterov)

    def optimize_epoch(self, model, criterion, optimizer, loader, epoch, is_validation=False):
        if is_validation is False:
            model.train()
            lr = self.lr_schedule(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            model.eval()

        for iter, (input, label) in enumerate(loader):
            input = input.to('cuda:0')
            label = label.to('cuda:0')
            predictions = model(input)
            for prediction in predictions:
                loss = criterion(prediction, label)
                if is_validation is False:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return


arch = 'alexnet'
data = 'Imagenet'
ckpt_dir = './test'
device = "1"
modelpath = '.ckpt400.pth'
results = ''
workers = 6
epochs = 36
batch_size = 192
learning_rate = 0.01
tencrops = False
evaluate = False
datadir = '/home/ubuntu/data/imagenet'
name = 'eval'

if __name__ == "__main__":
    print("=" * 60)
    print()
    util.setup_runtime(seed=2, cuda_dev_id=device)

    print(f"Training architecture {arch} on {data}")
    writer = SummaryWriter(f'./runs_LP/{data}/{name}')
    writer.add_text('args', " \n".join([f'{key}: {value}' for key, value in locals().items()]))

    model = model_with_probes(model_path=modelpath, which=data)
    train_loader, val_loader = get_standard_data_loader_pairs(dir_path=datadir,
                                                              batch_size=batch_size,
                                                              num_workers=workers,
                                                              tencrops=tencrops)

    o = LinearProbesOptimizer()
    o.lr = learning_rate
    o.validate_only = evaluate
    o.num_epochs = epochs
    o.checkpoint_dir = ckpt_dir
    o.resume = True
    o.optimize(model, train_loader, val_loader)
