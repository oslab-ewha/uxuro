function _get_bench_elapsed_time()
{
    sum=0
    elapsed_all=""
    for i in `seq 1 $n_benches`
    do
        str=`$* 2> /dev/null | grep "^elapsed:" | grep -Ewo '[[:digit:]]*\.[[:digit:]]*'`
        if [ $? -ne 0 ]; then
            echo -n "- -"
            return
        else
            sum=`echo "scale=6;$sum + $str" | bc`
	    elapsed_all="$elapsed_all $str"
        fi
	sleep 1
    done

    avg=`echo "scale=3; $sum / $n_benches" | bc`

    sum2=0
    for elapsed in $elapsed_all
    do
        sum2=`echo "scale=6;$sum2 + ($avg - $elapsed) * ($avg - $elapsed)" | bc`
    done

    avg2=`echo "scale=6;$sum2 / $n_benches" | bc`
    std=`echo "scale=3;sqrt($avg2)/1" | bc`

    echo -n $avg $std
}

n_benches=${n_benches:-5}

get_bench_elapsed_no_newline()
{
    _get_bench_elapsed_time $*
}

get_bench_elapsed()
{
    _get_bench_elapsed_time $*
    echo ""
}
