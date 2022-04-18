import csv


class DataBase:
    def __init__(self, paths):
        self.data = []

        fidx = 0
        for path in paths:
            self._load(fidx, path)
            fidx += 1

    def _load(self, fidx, path):
        try:
            f = open(path, 'r')
        except IOError:
            return
        self.reader = csv.reader(f, delimiter=',')

        evict_addr_min = None
        evict_addr_max = None
        ts_min = None
        ts_max = None

        for row in self.reader:
            self.data.append(self._parse_row(fidx, row))

    def _parse_row(self, fidx, row):
        pass

    def rebase_min(self, idx):
        minval = None

        for d in self.data:
            if not minval or minval > d[idx]:
                minval = d[idx]
        for i in range(len(self.data)):
            self.data[i][idx] = self.data[i][idx] - minval

    def get_plot_data(self, idx_x, idx_y):
        xs = []
        ys = []

        for d in self.data:
            xs.append(d[idx_x])
            ys.append(d[idx_y])
        return xs, ys
