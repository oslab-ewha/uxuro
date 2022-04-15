#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import data_evicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs="+", type=str, help='path of evicts kernel message file')
    parser.add_argument('-p', type=bool, default="", help='plot with points')
    parser.add_argument('-o', type=str, default="", help='output filename')
    args = parser.parse_args()

    data = data_evicts.DataEvicts(args.csv[0])

    marker = ',' if args.p else '.'

    fig = plt.figure()

    xs, ys = data.get_plot_data()
    plt.plot(xs, ys, marker=marker, ls='')
    plt.xlabel("Timestamp")
    plt.ylabel("Eviction Address")

    plt.tight_layout()

    if args.o:
        print('saving figure:', args.o)
        fig.savefig(args.o, dpi=500)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
