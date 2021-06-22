#!/bin/bash

trap 'cleanup' INT TERM

if [[ -z $DATA_BASE ]]; then
    DATA_BASE=/data
fi

output=.output.$$
no_drop_caches=

cleanup() {
    rm -f $output
}

run_as_tabular_output() {
    str=`$cmd $datadir 2> /dev/null | grep "time(us):" | grep -Ewo '[[:digit:]]*'`
    echo $CUIO_TYPE $str >> $output
}

drop_caches() {
	sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'
}

setup_runtime_by_code() {
    datadir=$DATA_BASE/data.$benchcode.$memscope.$size

    case $benchcode in
	bs)
	    cmd=./BlackScholes
	    ;;
	bp)
	    cmd=./backprop
	    ;;
	hs)
	    cmd=./hotspot
	    ;;
	lm)
	    cmd=./lavaMD
	    ;;
	va)
	    cmd=./vectorAdd
	    ;;
	pf)
	    cmd=./pathfinder
	    ;;
	*)
	    echo "unsupported code: $benchcode"
	    exit 2
	    ;;
    esac
}

setup_run_schemes() {
    if [[ $schemes = "all" ]]; then
	case $memscope in
	    fm)
		run_schemes="NVMGPU"
		;;
	    hm)
		run_schemes="UVM NVMGPU HOSTREG"
		;;
	    gm)
		run_schemes="UVM NVMGPU HOSTREG HOST"
		;;
	esac
    else
	run_schemes=$schemes
    fi
}

run_multi() {
    cnt=$1
    for i in $(seq 1 $cnt)
    do
	echo -n "[$CUIO_TYPE.$memscope.$size] running..... $i"
	run_as_tabular_output
	echo " ..Done"
	if [[ -z $no_drop_caches ]]; then
	    drop_caches
	fi
    done
}

add_banner_to_output() {
    echo -n "#cnt:$cnt,$schemes,$benchcode,$memscope,size:$size" >> $output
    if [[ -n $no_drop_caches ]]; then
	echo -n ",no_drop_caches" >> $output
    fi
    echo >> $output
}

run_schemes() {
    cnt=$1
    schemes=$2
    benchcode=$3
    memscope=$4
    size=$5

    add_banner_to_output

    setup_runtime_by_code
    setup_run_schemes

    for s in $run_schemes
    do
	export CUIO_TYPE=$s
	run_multi $cnt
    done

    echo >> $output
}

run_by_schemes_size() {
    run_schemes $*

    cat $output
    cleanup
}

run_by_all_sizes() {
    for i in `ls -d $DATA_BASE/data.$3.$4.? | grep -ow [[:digit:]]$`
    do
	run_schemes $* $i
    done

    cat $output
    cleanup
}

#
#argument: count, schemes, bench code(bs,bp), memory scope(fm,hm,gm)[, size(1~n) ]
#
run() {
    if [[ -z $5 ]]; then
	run_by_all_sizes $1 $2 $3 $4
    else
	run_by_schemes_size $1 $2 $3 $4 $5
    fi
}
