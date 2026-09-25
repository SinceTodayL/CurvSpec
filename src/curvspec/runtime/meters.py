class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count
