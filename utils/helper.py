import shutil
import torch


class AverageMeter(object):
    """Computes and stores the average, current value, and standard deviation, along with all values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []  # Lista para almacenar todos los valores
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sq_sum = 0  # Suma de cuadrados
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.values.extend([val] * n)  # Añade el valor 'n' veces a la lista de valores
        self.sum += val * n
        self.sq_sum += val**2 * n  # Actualizar la suma de cuadrados
        self.count += n
        self.avg = self.sum / self.count
        self.std = ((self.sq_sum / self.count) - (self.avg ** 2)) ** 0.5


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weights/best_check.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))#init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr