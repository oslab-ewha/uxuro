import data


class DataEvicts(data.DataBase):
    def __init__(self, paths):
        super().__init__(paths)
        self.rebase_min(0)
        self.rebase_min(1)

    def _parse_row(self, fidx, row):
        evict_addr = int(row[1], 16)
        ts = int(row[0], 16)
        return [ts, evict_addr]
