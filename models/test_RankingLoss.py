from unittest import TestCase

from models.RankingLoss import RankingLoss
import torch


class TestRankingLoss(TestCase):
    def test_forward(self):
        loss = RankingLoss()
        logits = torch.tensor([1., 2.])
        labels = torch.tensor([0.1, 0.2])

        excepted_loss = torch.tensor([2. - 1.]).sigmoid().log()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)

        labels = torch.tensor([0.2, 0.1])
        excepted_loss = torch.tensor([1. - 2.]).sigmoid().log()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)


if __name__ == '__main__':
    TestRankingLoss().test_forward()
