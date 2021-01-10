
import collections

class FuckBuffer:
    def __init__(self, queue_size=10, gamma=0.5):
        self.count = 0
        self.queue_size = queue_size
        self.gamma = gamma
        self.sum = 0.0
        self.wsum = 0.0
        self.c = collections.deque()

    def append(self, e):
        e = float(e)

        self.c.append(e)
        self.count += 1
        self.sum += e
        self.wsum = self.gamma * self.wsum + e

        if self.count > self.queue_size:
            val = self.c.popleft()
            self.sum -= val
            self.wsum -= val * self.gamma**(self.count+1)
            self.count -= 1

    def pop(self):
        self.count -= 1
        self.sum -= self.c.popleft()

    def avg(self):
        return self.sum / self.count

    def wavg(self):
        # if self.count:
        #     return self.wsum ** (1/self.count)
        total_weight = (1 - self.gamma ** (self.count + 1)) / (1 - self.gamma)
        return self.wsum / total_weight

    def clear(self):
        self.count = 0
        self.wsum = self.sum = 0.0
        self.c.clear()

