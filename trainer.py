import torch
import torch.nn.functional as F
from torch.autograd import Variable
from distutils.version import LooseVersion


def cross_entropy2d(pred, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, c, h, w)
    n, c, h, w = pred.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(pred)
    else:
        # >=0.3
        log_p = F.log_softmax(pred, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    target = target.transpose(1, 2).transpose(2, 3).contiguous()
    target = target.view(n, -1)

    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= (n*h*w)
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, epochs,
                 size_average=False):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.size_average = size_average

        self.epoch = 0
        self.iteration = 0

    def train_epoch(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            # optimizer step
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            loss.backward()
            self.optim.step()
            if iteration % 5 == 0:
                print('iteration:{} loss:{:.4f}'.format(iteration, loss_data))
        train_acc = self.validate(self.train_loader)
        test_acc = self.validate(self.test_loader)
        print('train accuracy:{:.4f} test accuracy:{:.4f}'.format(
            train_acc, test_acc))

    def train(self):

        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()

    def validate(self, data_loader):
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            accuracy = 0
            total_num = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):

                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optim.zero_grad()
                # score: [n, c, h, w]
                score = self.model(data)
                pred = F.softmax(score, dim=1).max(1)[1]
                accuracy += torch.eq(pred, target).float().mean()
                total_num += data.shape[0]
            avg_acc = accuracy / total_num

        if training:
            self.model.train()

        return avg_acc.data.item
