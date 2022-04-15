import csv


class DataEvicts:
    def __init__(self, path):
        self.data = []
        self._load(path)

    def _load(self, path):
        try:
            f = open(path, 'r')
        except IOError:
            return
        reader = csv.reader(f, delimiter=',')

        evict_addr_min = None
        evict_addr_max = None
        ts_min = None
        ts_max = None

        for row in reader:
            evict_addr = int(row[1], 16)
            ts = int(row[0], 16)
            if not evict_addr_min or evict_addr_min > evict_addr:
                evict_addr_min = evict_addr
            if not evict_addr_max or evict_addr_max < evict_addr:
                evict_addr_max = evict_addr
            if not ts_min or ts_min > ts:
                ts_min = ts
            if not ts_max or ts_max < ts:
                ts_max = ts

            self.data.append((ts, evict_addr))

        for i in range(len(self.data)):
            self.data[i] = (self.data[i][0] - ts_min, self.data[i][1] - evict_addr_min)

    def get_plot_data(self):
        timestamps = []
        evict_addrs = []

        for d in self.data:
            timestamps.append(d[0])
            evict_addrs.append(d[1])
        return timestamps, evict_addrs
