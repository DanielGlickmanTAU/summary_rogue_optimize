from torch.nn import MSELoss, BCEWithLogitsLoss

from config.config import RankerConfig
from models.loss.MeanCenteredMSELoss import MeanCenteredMSELoss
from models.loss.NormalizedMSELoss import NormalizedMSELoss
from models.loss.RankNetLoss import RankNetLoss
from models.loss.RankingLoss import RankingLoss


def get_loss(config: RankerConfig):
    loss_fn = config['loss_fn'] if isinstance(config, dict) else config.loss_fn

    if loss_fn == 'bce':
        return BCEWithLogitsLoss()
    if loss_fn == 'centered-mse':
        return MeanCenteredMSELoss(reduction='mean')
    if loss_fn == 'normalized-mse':
        return NormalizedMSELoss(reduction='mean')
    if loss_fn == 'mse':
        return MSELoss(reduction='mean')
    if loss_fn == 'ranking':
        tolerance = config.tolerance
        return RankingLoss(tolerance=tolerance, reduction='mean')
    if loss_fn == 'rank-net':
        return RankNetLoss()
    raise Exception(f'unknown {loss_fn} ')
