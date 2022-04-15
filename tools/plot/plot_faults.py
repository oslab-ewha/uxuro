#!/usr/bin/python3

import argparse
import matplotlib.pyplot as plt
import data_faults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs="+", type=str, help='path of fault kernel message file')
    parser.add_argument('-c', type=str, default="",
                        help='fault category. all:rw+fault, rw:read/write fault, fault: fault type')
    parser.add_argument('-p', type=bool, default="", help='plot with points')
    parser.add_argument('-o', type=str, default="", help='output filename')
    args = parser.parse_args()

    data = data_faults.DataFaults(args.csv[0])

    marker = ',' if args.p else '.'

    fig = plt.figure()
    if args.c == 'all':
        xs, ys = data.get_plot_data(ftype=0, acctype=1)
        plt.plot(xs, ys, marker=marker, label="PDE/R", ls='')
        xs, ys = data.get_plot_data(ftype=0, acctype=2)
        plt.plot(xs, ys, marker=marker, label="PDE/W", ls='')
        xs, ys = data.get_plot_data(ftype=1, acctype=1)
        plt.plot(xs, ys, marker=marker, label="PTE/R", ls='')
        xs, ys = data.get_plot_data(ftype=1, acctype=2)
        plt.plot(xs, ys, marker=marker, label="PTE/W", ls='')
    elif args.c == 'rw':
        xs, ys = data.get_plot_data(acctype=1)
        plt.plot(xs, ys, marker=marker, label="Read", ls='')
        xs, ys = data.get_plot_data(acctype=2)
        plt.plot(xs, ys, marker=marker, label="Write", ls='')
    elif args.c == 'fault':
        xs, ys = data.get_plot_data(ftype=0)
        plt.plot(xs, ys, marker=marker, label="PDE", ls='')
        xs, ys = data.get_plot_data(ftype=1)
        plt.plot(xs, ys, marker=marker, label="PTE", ls='')
    else:
        xs, ys = data.get_plot_data()
        plt.plot(xs, ys, marker=marker, ls='')

    plt.xlabel("Timestamp")
    plt.ylabel("Faulting Address")
    if args.c:
        plt.legend()

    plt.tight_layout()

    if args.o:
        print('saving figure:', args.o)
        fig.savefig(args.o, dpi=500)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
