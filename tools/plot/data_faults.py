import csv


class DataFaults:
    def __init__(self, path):
        self.data = []
        self._load(path)

    def _load(self, path):
        try:
            f = open(path, 'r')
        except IOError:
            return
        reader = csv.reader(f, delimiter=',')

        fault_addr_min = None
        fault_addr_max = None
        ts_min = None
        ts_max = None

        for row in reader:
            fault_addr = int(row[0], 16)
            ts = int(row[1])
            ftype = int(row[2])
            acctype = int(row[3])
            if not fault_addr_min or fault_addr_min > fault_addr:
                fault_addr_min = fault_addr
            if not fault_addr_max or fault_addr_max < fault_addr:
                fault_addr_max = fault_addr
            if not ts_min or ts_min > ts:
                ts_min = ts
            if not ts_max or ts_max < ts:
                ts_max = ts

            self.data.append((ts, fault_addr, ftype, acctype))

        for i in range(len(self.data)):
            self.data[i] = (self.data[i][0] - ts_min, self.data[i][1] - fault_addr_min, self.data[i][2], self.data[i][3])

    def get_plot_data(self, **kwargs):
        timestamps = []
        fault_addrs = []

        ftype = None
        acctype = None
        if 'ftype' in kwargs:
            ftype = kwargs['ftype']
        if 'acctype' in kwargs:
            acctype = kwargs['acctype']
        for d in self.data:
            if ftype and ftype != d[2]:
                continue
            if acctype and acctype != d[3]:
                continue
            timestamps.append(d[0])
            fault_addrs.append(d[1])
        return timestamps, fault_addrs
