import argparse
import matplotlib.pyplot as plt


class PlotBase:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('csv', nargs="+", type=str, help='path of kernel message file')
        self.parser.add_argument('--points', action='store_true', help='plot with points')
        self.parser.add_argument('-o', type=str, default="", help='output filename')
        self.parser.set_defaults(points=False)
        self._add_args()
        self.args = self.parser.parse_args()
        self.marker = ',' if self.args.points else '.'

        self.fig = plt.figure()

    def show(self):
        plt.tight_layout()

        if self.args.o:
            print('saving figure:', self.args.o)
            self.fig.savefig(args.o, dpi=500)
            plt.close(self.fig)
        else:
            plt.show()

    def _add_args(self):
        pass