import csv


class DataBase:
    def __init__(self, path, head_ch=None):
        self.data = []

        self._load(path, head_ch)

    def _load(self, path, head_ch):
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
            if head_ch and row[0] != head_ch:
                continue
            row_parsed = self._parse_row(row)
            if row_parsed:
                self.data.append(row_parsed)

    def _parse_row(self, row):
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
