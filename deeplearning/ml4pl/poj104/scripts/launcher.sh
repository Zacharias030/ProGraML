#!/bin/bash

# Require a directory
if [[ $# -ge 1 ]]; then
    cd "$1"
else
    echo 'Specify a directory to launch jobs. exiting.'
    exit 1
fi

# ask for launching any .sh file.
for file in *.sh; do
    export APP=$file
    read -n1 -p $"$file -- Submit this job [y,n]:" doit
    case $doit in
      y|Y) echo -e "\033[2K"
       sbatch -n 1 < $file;
        echo ----- Submitted $file ----- ;;
      n|N) echo -e "\033[2K"
        echo skipping $file... ;;
      *) echo Unknown choice. skipping $file... ;;
    esac
done
