from unittest import TestCase

import torch

from train import training


class Test(TestCase):
    def test_best_at_k(self):
        labels = [
            [0.0000, 0.1481],
            [0.6957, 0.6818],
            [0.2927, 0.4444],
            [0.2286, 0.1143]
        ]

        logits = [
            [0.1000, 0.2],
            [0.693, 0.6848],
            [0.2927, 0.4444],
            [0.2286, 0.1143]
        ]

        best, _ = training.best_at_k(torch.tensor(labels), torch.tensor(logits), k=2)

        self.assertAlmostEqual(best, (0.1481 + 0.6957 + 0.4444 + 0.2286) / 4)

        labels = torch.tensor([[0.2286, 0.1143],
                               [0.0000, 0.1481]])

        logits = torch.tensor([[[0.2349],
                                [0.1182]],
                               [[0.0050],
                                [0.1552]]])
        best, _ = training.best_at_k(labels, logits, k=2)

        self.assertAlmostEqual(best, (0.1481 + 0.2286) / 2)
