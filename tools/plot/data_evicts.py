import data


class DataEvicts(data.DataBase):
    def __init__(self, path):
        super().__init__(path, 'e')
        self.rebase_min(0)
        self.rebase_min(1)

    def _parse_row(self, row):
        evict_addr = int(row[2], 16)
        ts = int(row[1], 16)
        return [ts, evict_addr]
