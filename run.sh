#!/bin/bash
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

algos=$1  # qmix
envs=$2     # sc2
args=$3    # use_cuda=True
threads=$4 # 2
gpus=$5    # 0,1
times=$6   # 5

algos=(${algos//,/ })
envs=(${envs//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })

if [ ! $algos ] || [ ! $envs ] ; then
    echo "Please enter the correct command."
    echo "bash run_fifo.sh algo_name_list env_name_list arg_list experiment_thread_num gpu_list experiment_num"
    exit 1
fi

if [ ! $threads ]; then
  threads=1
fi

if [ ! $gpus ]; then
  gpus=(0)
fi

if [ ! $times ]; then
  times=6
fi

echo "ALGO LIST:" ${algos[@]}
echo "ENV LIST:" envs
echo "ARGS:"  ${args[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times


# fifo
# https://www.cnblogs.com/maxgongzuo/p/6414376.html
FIFO_FILE=$(mktemp)
rm $FIFO_FILE
mkfifo $FIFO_FILE
trap "rm $FIFO_FILE" 3
trap "rm $FIFO_FILE" 15

exec 6<>$FIFO_FILE

for ((idx=0;idx<threads;idx++)); do
    echo
done >&6


# run parallel
count=0
for algo in "${algos[@]}"; do
  for env in "${envs[@]}"; do
      for((i=0;i<times;i++)); do
          read -u6
          gpu=${gpus[$(($count % ${#gpus[@]}))]}
          {
#            CUDA_VISIBLE_DEVICES="$gpu" python main.py --env="$env" "${args[@]}" --seed="$i"
            CUDA_VISIBLE_DEVICES="$gpu" python main.py --env="$env" --seed="$i"
            echo >&6
          } &
          count=$(($count + 1))
          sleep $((RANDOM % 60 + 60))
      done
  done
done
wait

exec 6>&-   # 关闭fd6
rm $FIFO_FILE
