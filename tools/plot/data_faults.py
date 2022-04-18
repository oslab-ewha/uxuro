import data


class DataFaults(data.DataBase):
    def __init__(self, paths):
        super().__init__(paths)
        self.rebase_min(0)
        self.rebase_min(1)

    def _parse_row(self, fidx, row):
        fault_addr = int(row[1], 16)
        ts_fault = int(row[2])
        ftype = int(row[3])
        acctype = int(row[4])
        return [ts_fault, fault_addr, ftype, acctype]

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
