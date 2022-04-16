#!/usr/bin/python3

import data_evicts
import plot
import matplotlib.pyplot as plt


class PlotEvicts(plot.PlotBase):
    def __init__(self):
        super().__init__()

    def show(self):
        data = data_evicts.DataEvicts(self.args.csv[0])

        xs, ys = data.get_plot_data()
        plt.plot(xs, ys, marker=self.marker, ls='')
        plt.xlabel("Timestamp")
        plt.ylabel("Eviction Address")

        super().show()


if __name__ == "__main__":
    plt_evicts = PlotEvicts()
    plt_evicts.show()
