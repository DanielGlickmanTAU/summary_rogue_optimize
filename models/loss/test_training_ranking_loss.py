from unittest import TestCase

from train.training import best_at_k
from utils import compute

from torch import optim
from torch.nn import MSELoss

from models.loss.RankingLoss import RankingLoss
import torch

device = torch.device('cuda')


class TrainingTester(TestCase):
    def test_train(self):
        loss_fn = RankingLoss()
        # loss_fn = MSELoss()
        ff = self.get_ff().to(device)
        X, Y = self.get_dataset(num_samples=1, candidates_per_sample=4)
        X, Y = X.to(device), Y.to(device)

        optimizer = optim.SGD(ff.parameters(), lr=2e-3)
        epochs = 100
        X = X.view((-1, 4, 3))
        Y = labels_tensor = Y.view((-1, 4))
        print('oracle', best_at_k(labels_tensor, labels_tensor, k=4))
        for i in range(epochs):
            optimizer.zero_grad()
            output = ff(X)
            loss = loss_fn(output.view(Y.shape), Y)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                # print(X, output)
                print('step', i, 'loss', loss)
                print(best_at_k(labels_tensor, output, k=4))

    def test_get_dataset(self):
        x, y, = self.get_dataset(2, 10, 3)
        assert x.shape == (2, 10, 3)
        assert y.shape == (2, 10)
        first_example = x[0]
        first_example_labels = y[0]
        first_candidate = first_example[0]

        assert first_example_labels[0] == first_candidate @ torch.tensor([4., 2., 1])

    def get_ff(self, hidden=10):
        return torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device)

    def get_dataset(self, num_samples=1_000, candidates_per_sample=4, dim=3):
        X = torch.randn(num_samples, candidates_per_sample, dim)
        Y = X @ torch.tensor([4., 2., 1])
        return X, Y
