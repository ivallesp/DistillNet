#!/bin/bash
set -e


FPB_LAST_TS=$(date +%s%N | cut -b1-10)
FPB_TIME_PER_STEP=-1

calc() { awk "BEGIN{print $*}"; }

calc_int() { awk "BEGIN{print int($*)}"; }

fancy_pb() {
    i=$1  # Iteration number
    n_its=$2  # Total number of iterations
    print_every=$3  # Print frequency
    size=$4  # Width of the progress bar in number of characters
    msg=${5:-''}
    # Initialize the last timestamp in the first iteration
    [ $i -eq 1 ] && FPB_LAST_TS=$(date +%s%N | cut -b1-10)
    ALPHA=0.1 # Exponential smoothing parameter
    if !(($i % $print_every)); then
        CURRENT_MILLIS=$(date +%s%N | cut -b1-10)
        if [ $i -le 2 ]; then
            # In the first iteration, get the point estimate
            FPB_TIME_PER_STEP=$( calc "$CURRENT_MILLIS - $FPB_LAST_TS" )
        else
            # In further iterations, apply exponential smoothing
            FPB_TIME_PER_STEP=$( calc "$ALPHA * ($CURRENT_MILLIS - $FPB_LAST_TS) + (1 - $ALPHA) * $FPB_TIME_PER_STEP" )
        fi
        # Calculate the ETA estimate
        eta=$( calc_int "$FPB_TIME_PER_STEP * ( $n_its - $i ) / $print_every" )

        # Draw the progress bar bar
        echo -n "[" # Bar frame left edge
        for ((j=0; j<$(expr $i '*' $size / $n_its); j++)) ; do echo -n ' '; done # Spaces pre cursor
        echo -n '=>' # Cursor
        for ((j=$(expr $i '*' $size / $n_its); j<$size; j++)) ; do echo -n ' '; done # Spaces post cursor
        # Print the stats
        echo -n ] $i / $n_its - ETA: $(show_time $eta) - $msg $'\r'
        # Update the last timestamp variable
        FPB_LAST_TS=$CURRENT_MILLIS
    fi
}

function show_time () {
    num=$1
    min=0
    hour=0
    day=0
    if((num>59));then
        ((sec=num%60))
        ((num=num/60))
        if((num>59));then
            ((min=num%60))
            ((num=num/60))
            if((num>23));then
                ((hour=num%24))
                ((day=num/24))
            else
                ((hour=num))
            fi
        else
            ((min=num))
        fi
    else
        ((sec=num))
    fi
    echo "$day"d "$hour"h "$min"m "$sec"s
}

i=1
n_files=$(find models/ -type f -iname '*.h5' | wc -l)
for F in $(find models/ -type f -iname '*.h5' | shuf)
do
    fancy_pb $i $n_files 1 20 $F
    python infer.py -m $F
    ((i++))
done