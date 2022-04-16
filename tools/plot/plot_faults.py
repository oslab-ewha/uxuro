#!/usr/bin/python3

import matplotlib.pyplot as plt
import data_faults
import plot


class PlotFaults(plot.PlotBase):
    def __init__(self):
        super().__init__()

    def _add_args(self):
        self.parser.add_argument('-c', type=str, default="",
                                 help='fault category. all:rw+fault, rw:read/write fault, fault: fault type')

    def show(self):
        data = data_faults.DataFaults(self.args.csv[0])

        if self.args.c == 'all':
            xs, ys = data.get_plot_data(ftype=0, acctype=1)
            plt.plot(xs, ys, marker=self.marker, label="PDE/R", ls='')
            xs, ys = data.get_plot_data(ftype=0, acctype=2)
            plt.plot(xs, ys, marker=self.marker, label="PDE/W", ls='')
            xs, ys = data.get_plot_data(ftype=1, acctype=1)
            plt.plot(xs, ys, marker=self.marker, label="PTE/R", ls='')
            xs, ys = data.get_plot_data(ftype=1, acctype=2)
            plt.plot(xs, ys, marker=self.marker, label="PTE/W", ls='')
        elif self.args.c == 'rw':
            xs, ys = data.get_plot_data(acctype=1)
            plt.plot(xs, ys, marker=self.marker, label="Read", ls='')
            xs, ys = data.get_plot_data(acctype=2)
            plt.plot(xs, ys, marker=self.marker, label="Write", ls='')
        elif self.args.c == 'fault':
            xs, ys = data.get_plot_data(ftype=0)
            plt.plot(xs, ys, marker=self.marker, label="PDE", ls='')
            xs, ys = data.get_plot_data(ftype=1)
            plt.plot(xs, ys, marker=self.marker, label="PTE", ls='')
        else:
            xs, ys = data.get_plot_data()
            plt.plot(xs, ys, marker=self.marker, ls='')

        plt.xlabel("Timestamp")
        plt.ylabel("Faulting Address")
        if self.args.c:
            plt.legend()

        super().show()


if __name__ == "__main__":
    plt_faults = PlotFaults()
    plt_faults.show()
