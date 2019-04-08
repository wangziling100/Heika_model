import torch
import pandas as pd
import pickle
import numpy as np


class MMGRM():
    def __init__(self, n_items, n_tasks, cl, start_grade, regular=True):
        """
        structure of data and parameters
                tasks
        items   performance

        """
        #self.parameters = torch.nn.Parameter()
        self.parameters = []
        self.n_items = n_items
        self.n_tasks = n_tasks
        self.start = start_grade

        self.alpha = torch.rand(1, self.n_items, requires_grad=True)
        self.b = torch.rand(1, self.n_items, requires_grad=True)
        self.w = torch.rand(1, self.n_items, requires_grad=True)
        self.alpha = torch.nn.Parameter(self.alpha)
        self.b = torch.nn.Parameter(self.b)
        self.w = torch.nn.Parameter(self.w)
        self.parameters = [self.alpha, self.b, self.w]
        #self.parameters = [self.alpha, self.w]
        self.cl = cl
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.regular_fn = torch.nn.LogSigmoid()
        self.regular = regular

    def LoadLabels(self, labels, mask, test_ratio=0.1):
        n_test = int(n_tasks*test_ratio)
        indices = np.random.permutation(labels.shape[0])
        test_idx, train_idx = indices[:n_test], indices[n_test:]
        self.train_labels, self.test_labels = labels[train_idx,:], labels[test_idx,:]
        self.train_labels = torch.Tensor(self.train_labels)
        self.test_labels = torch.Tensor(self.test_labels)
        self.train_mask, self.test_mask = mask[train_idx,:], mask[test_idx,:]
        self.train_mask = torch.Tensor(self.train_mask)
        self.test_mask = torch.Tensor(self.test_mask)

        self.theta = torch.rand(self.train_labels.shape[0], 1, requires_grad=True)
        self.theta = torch.nn.Parameter(self.theta)
        self.parameters.append(self.theta)

    def SCF(self, c):
        tmp = self.alpha*(self.w*self.theta-self.b+c)
        return torch.exp(tmp)/(1+torch.exp(tmp))

    def MGRM(self, c):
        tmp = self.alpha*(self.theta-self.b+c)
        return torch.exp(tmp)/(1+torch.exp(tmp))

    def ESCF(self):
        s = self.start
        for c in self.cl:
            s += self.SCF(c)
        return s
    
    def ESCF_MGRM(self):
        s = self.start
        for c in self.cl:
            s += self.MGRM(c)
        return s

    def masked_ESCF(self, data_set='train'):
        if data_set == 'train':
            return self.ESCF()*self.train_mask
        elif data_set == 'test':
            raise NotImplementedError

    def masked_ESCF_MGRM(self, data_set='train'):
        if data_set == 'train':
            return self.ESCF_MGRM()*self.train_mask
        elif data_set == 'test':
            raise NotImplementedError

    def add_regular_term(self, data_set='train'):
        if data_set == 'train':
            return self.masked_ESCF()-self.regular_fn(self.alpha)-self.regular_fn(self.w)
        elif data_set == 'test':
            raise NotImplementedError

    def loss(self):
        # return self.loss_fn(self.masked_ESCF_MGRM(), self.train_labels)
        if self.regular:
            return self.loss_fn(self.add_regular_term(), self.train_labels)
        else:
            return self.loss_fn(self.masked_ESCF(), self.train_labels)

    def loss2(self):
        return self.loss_fn(self.masked_ESCF(), self.train_labels)

    def train(self, loops=10000, lr=1e-3, opt='adam'):
        if opt == 'adam':
            print(self.parameters)
            self.opt = torch.optim.Adam(self.parameters, lr=lr)

        for t in range(loops):
            loss = self.loss()
            print(t, loss.item())
            print(t, self.loss2().item())
            self.opt.zero_grad()

            loss.backward()
            self.opt.step()

    def store_params(self, fn, store_theta=False):
        alpha = self.alpha.detach().numpy()
        alpha = pd.Series(alpha.reshape(-1))
        b = self.b.detach().numpy()
        b = pd.Series(b.reshape(-1))
        w = self.w.detach().numpy()
        w = pd.Series(w.reshape(-1))

        df = pd.concat([alpha, b, w], axis=1)
        df.columns = ['alpha', 'b', 'w']
        df['difficulty'] = 1
        df['difficulty'] = df['difficulty'].cumsum()

        if store_theta:
            raise NotImplementedError
        df.to_csv(fn, index=False)


def set_cl(n=19, start=-1, end=1, test_col_p=None):
    if start < end:
        tmp = start
        start = end
        end = tmp
    if test_col_p is None or test_col_p is False:
        return list(np.linspace(start, end, n))
    else:
        test_col_p = test_col_p[::-1]
        tmp = pd.Series(test_col_p)
        mx = tmp.max()
        mn = tmp.min()
        tmp = (tmp-mn)/(mx-mn)
        tmp = tmp*(end-start)+start
        return tmp.tolist()


if __name__ == '__main__':
    data = pd.read_csv('workspace/data/step2_expected_performance.csv')

    divide_points = None
    with open('workspace/data/ability_mapper.p', 'rb') as f:
        mapping_parameters = pickle.load(f)
    if len(mapping_parameters) == 7:
        col, how, n_level, invert, v_max, v_min, divide_points = mapping_parameters
    elif len(mapping_parameters) == 9:
        col, how, n_level, invert, v_max, v_min, divide_points, balanced_scale, avg_perf = mapping_parameters
    min_grade = data['performance'].min()
    print(data.columns)

    cl = set_cl(n_level, test_col_p=divide_points)
    print('cl:', cl)

    n_items = len(data['difficulty'].unique())
    n_tasks = len(data.groupby(['uid', 'day', 'exc_num', 'exc_times']).mean())

    print('# of items:', n_items)
    print('# of tasks:', n_tasks)
    model = MMGRM(n_items, n_tasks, cl, min_grade)

    data = data.groupby(['uid', 'day', 'exc_num', 'exc_times', 'difficulty'])['performance'].mean().reset_index()
    data = data.set_index(['uid', 'day', 'exc_num', 'exc_times'])
    data = data.pivot_table(values='performance', index=data.index, columns='difficulty', aggfunc='first')
    mask = data.values
    y = np.nan_to_num(mask)
    mask = mask/mask
    mask = np.nan_to_num(mask)
    model.LoadLabels(y, mask)
    model.train(loops=2000)
    model.store_params('workspace/data/item_params.csv')
    
    pass






