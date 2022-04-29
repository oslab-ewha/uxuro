import data


class DataBatches(data.DataBase):
    def __init__(self, path):
        super().__init__(path, 'b')
        self.rebase_min(0)

    def _parse_row(self, row):
        ts = row[1]
        n_faults = int(row[2])
        n_coalesced_faults = int(row[3])
        n_dup_faults = int(row[4])
        n_invalid_prefetch_faults = int(row[5])

        return [ts, n_faults, n_coalesced_faults, n_dup_faults, n_invalid_prefetch_faults]
