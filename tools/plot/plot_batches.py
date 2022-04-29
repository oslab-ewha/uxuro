#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

import data_batches
import plot


class PlotBatches(plot.PlotBase):
    def __init__(self):
        super().__init__()

    def _add_args(self):
        self.parser.add_argument('-c', type=str, default="fault",
                                 help='fault category: fault, coalesced, dup')

    def show(self):
        data = data_batches.DataBatches(self.args.csv[0])

        ts_min, ts_max = data.get_data_range(0)
        bar_width = (ts_max - ts_min) / 500

        data_idx = {'fault': 1, 'coalesced': 2, 'dup': 3}
        xs, ys = data.get_plot_data(0, data_idx[self.args.c])
        plt.bar(xs, ys, width=bar_width)

        plt.xlabel("Timestamp")
        plt.ylabel("# of faults")

        super().show()


if __name__ == "__main__":
    plt_batches = PlotBatches()
    plt_batches.show()
