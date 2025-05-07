#!/bin/bash

path_dst=$1


Sources=(
	Simulations/Max-Monitoring/Compares.png
        Simulations/Max-Monitoring/MinFg.png
        Simulations/Max-Monitoring/MinTime.png

)


Names=(
	Fig1_A.png
	Fig1_B.png
	Fig1_C.png
)





for i in "${!Sources[@]}"; do
    basename "${Sources[$i]}"
    f="${Names[$i]}"
    echo $filename
    file_dst="${path_dst}/${f}"

    echo $file_dst

    cp "${Sources[$i]}" "$file_dst"
    echo cp "${Sources[$i]}" "$file_dst"
done

