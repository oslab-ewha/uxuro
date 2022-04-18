import data


class DataEvictFaults(data.DataBase):
    def __init__(self, paths):
        super().__init__(paths)
        self.rebase_min(0)
        self.rebase_min(1)

    def _parse_row(self, fidx, row):
        ts = int(row[0], 16)
        if fidx == 0:
            fault_addr = int(row[1], 16)
            return [ts, fault_addr, False]
        else:
            evict_addr = int(row[1], 16)
            return [ts, evict_addr, True]

    def get_plot_data(self, is_evict: bool):
        timestamps = []
        addrs = []
        for d in self.data:
            if is_evict != d[2]:
                continue
            timestamps.append(d[0])
            addrs.append(d[1])
        return timestamps, addrs
