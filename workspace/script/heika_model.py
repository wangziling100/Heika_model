import torch
import pandas as pd


class MMGRM():
    def __init__(self, n_items, n_tasks, cl):
        self.n_items = n_items
        self.n_tasks = n_tasks

        self.alpha = torch.rand(self.n_items, 1, requires_grad=True)
        self.b = torch.rand(self.n_items, 1, requires_grad=True)
        self.w = torch.rand(self.n_items, 1, requires_grad=True)
        self.theta = torch.rand(1, self.n_tasks, requires_grad=True)
        self.cl = cl

    def LoadData(self, train=None, test=None):
        self.train = train
        self.test = test

    def SCF(self, c):
        tmp = self.alpha*(self.w*self.theta-self.b+c)
        return torch.exp(tmp)/(1+torch.exp(tmp))

    def ESCF(self, start):
        s = start
        for c in self.cl:
            s += self.SCF(c)
        return s


if __name__ == '__main__':
    pass






