#!/usr/bin/python3

import data_evict_faults

import plot
import matplotlib.pyplot as plt


class PlotEvictFaults(plot.PlotBase):
    def __init__(self):
        super().__init__()

    def show(self):
        data = data_evict_faults.DataEvictFaults(self.args.csv)

        xs_f, ys_f = data.get_plot_data(False)
        marker = 'x' if self.marker == '.' else ','
        plt.plot(xs_f, ys_f, marker=marker, label='fault', ls='')
        xs_e, ys_e = data.get_plot_data(True)
        marker = '+' if self.marker == '.' else ','
        plt.plot(xs_e, ys_e, marker=marker, label="evict", ls='')
        plt.xlabel("Timestamp")
        plt.ylabel("Fault/Eviction Address")
        plt.legend()

        super().show()


if __name__ == "__main__":
    plt_evictfaults = PlotEvictFaults()
    plt_evictfaults.show()
