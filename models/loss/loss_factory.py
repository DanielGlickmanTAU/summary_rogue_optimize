from torch.nn import MSELoss

from models.loss.MeanCenteredMSELoss import MeanCenteredMSELoss
from models.loss.NormalizedMSELoss import NormalizedMSELoss
from models.loss.RankNetLoss import RankNetLoss
from models.loss.RankingLoss import RankingLoss


def get_loss(config):
    loss_fn = config['loss_fn'] if isinstance(config, dict) else config.loss_fn
    if loss_fn == 'centered-mse':
        return MeanCenteredMSELoss(reduction='sum')
    if loss_fn == 'normalized-mse':
        return NormalizedMSELoss(reduction='sum')
    if loss_fn == 'mse':
        return MSELoss(reduction='sum')
    if loss_fn == 'ranking':
        tolerance = config.tolerance
        return RankingLoss(tolerance=tolerance, reduction='sum')
    if loss_fn == 'rank-net':
        return RankNetLoss()
    raise Exception(f'unknown {loss_fn} ')
