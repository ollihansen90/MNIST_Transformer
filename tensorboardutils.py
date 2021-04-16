
from torch.utils.tensorboard import SummaryWriter

class TBScalar:
    def __init__(self, parent_logger, name):
        self.logger = parent_logger
        self.name = name
    def append(self, value, global_step):
        self.logger.writer.add_scalar(self.name, value, global_step)
class TBLogger:
    def __init__(self, logdir="log"):
        self.writer = SummaryWriter(logdir)
    def new_scalar(self, name):
        return TBScalar(self, name)

"""  
logger = TBLogger(logdir)
architect_loss_log = logger.new_scalar('architect_loss')
architect_loss_log.append(architect_loss_accum, i * world_size)"""