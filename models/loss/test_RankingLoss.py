from unittest import TestCase

from models.loss.RankingLoss import RankingLoss
import torch


class TestRankingLoss(TestCase):
    def test_forward_simple(self):
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

    def test_forward_below_tolerance(self):
        tolerance = 0.01
        loss = RankingLoss(tolerance=tolerance)
        logits = torch.tensor([1., 2.])
        labels = torch.tensor([0.1, 0.1 + tolerance])

        excepted_loss = torch.tensor([0.])
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)

    def test_forward(self):
        loss = RankingLoss(tolerance=0., reduction='sum')
        labels = torch.tensor([3., 2., 1.])

        logits = torch.tensor([0.1, 0.2, 0.3])
        excepted_loss = torch.tensor([0.1 - 0.2, 0.1 - 0.3, 0.2 - 0.3]).sigmoid().log().sum()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)
        print(f'loss for logits {logits} ordered is {excepted_loss}')

        logits = torch.tensor([0.2, 0.1, 0.3])
        excepted_loss = torch.tensor([0.2 - 0.1, 0.2 - 0.3, 0.1 - 0.3]).sigmoid().log().sum()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)
        print(f'loss for logits {logits} ordered is {excepted_loss}')

        logits = torch.tensor([0.3, 0.2, 0.1])
        excepted_loss = torch.tensor([0.3 - 0.2, 0.3 - 0.1, 0.2 - 0.1]).sigmoid().log().sum()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)
        print(f'loss for logits {logits} ordered is {excepted_loss}')

    def test_forward_reordering(self):
        loss = RankingLoss(tolerance=0.1, reduction='sum')
        labels = torch.tensor([2., 1., 3.])
        logits = torch.tensor([0.1, 0.2, 0.3])
        excepted_loss = torch.tensor([0.3 - 0.1, 0.3 - 0.2, 0.1 - 0.2]).sigmoid().log().sum()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)

        labels = torch.tensor([2.001, 2., 3.])
        logits = torch.tensor([0.1, 0.2, 0.3])
        # third zeros out, because the diff in labels is < tolerance
        excepted_loss = torch.tensor([0.3 - 0.1, 0.3 - 0.2]).sigmoid().log().sum()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)

    def test_forward_multi_batch(self):
        loss = RankingLoss()

        logits = torch.tensor([[1., 2.], [1., 2.]])
        labels = torch.tensor([[0.1, 0.2], [0.2, 0.1]])

        excepted_loss1 = torch.tensor([2. - 1.]).sigmoid().log()
        excepted_loss2 = torch.tensor([1. - 2.]).sigmoid().log()
        excepted_loss = torch.tensor([excepted_loss1, excepted_loss2]).mean()
        real_loss = loss(logits, labels)
        self.assertAlmostEqual(real_loss.item(), excepted_loss.item(), delta=0.001)


if __name__ == '__main__':
    TestRankingLoss().test_forward_simple()
    TestRankingLoss().test_forward_below_tolerance()
    TestRankingLoss().test_forward()
    TestRankingLoss().test_forward_reordering()
    TestRankingLoss().test_forward_multi_batch()
