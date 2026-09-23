#!/usr/bin/env bash
# Usage: scripts/run_submission.sh <checkpoint> <name> "<gpu list>"
# Full Track 2 submission pipeline: one prediction worker per GPU (resumable), then check, then the official
# subsampling binaries. Raw predictions: submissions/track2/<name>/raw (~150-200 GB for 21 conditions, deletable
# after bundling); upload files: submissions/track2/<name>/bundle/{clean,robust}/*.hdf5.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."; W=${ROCO_TRACK2_WORK:-$PWD}
CK=$1; NAME=$2; GPUS=($3); N=${#GPUS[@]}
SUB=$W/submissions/track2/$NAME; RAW=$SUB/raw; LOG=$SUB/logs; ST=$W/outputs/track2/pipeline_status.txt
mkdir -p $RAW $LOG
echo "$(date '+%F %T') [submission] $NAME start: ckpt=$CK gpus=${GPUS[*]}" | tee -a $ST
free_kib=$(df -Pk "$W" | awk 'NR==2 {print $4}')
if (( free_kib < 250 * 1024 * 1024 )); then echo "less than 250 GiB free on $W; aborting" | tee -a $ST; exit 2; fi
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python3 -u scripts/make_submission.py predict --checkpoint $CK --root $RAW --worker $i/$N > $LOG/predict_w$i.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
# final sweep: fills anything skipped because of a stale lock (normally nothing)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 -u scripts/make_submission.py predict --checkpoint $CK --root $RAW --worker 0/1 --sweep > $LOG/predict_sweep.log 2>&1
find $RAW -name "*.lock" -delete
python3 -u scripts/make_submission.py check --root $RAW --deep > $LOG/check.log 2>&1 || { echo "$(date '+%F %T') [submission] $NAME CHECK FAILED (see $LOG/check.log)" | tee -a $ST; exit 1; }
python3 -u scripts/make_submission.py bundle --root $RAW --out $SUB/bundle > $LOG/bundle.log 2>&1 || { echo "$(date '+%F %T') [submission] $NAME BUNDLE FAILED (see $LOG/bundle.log)" | tee -a $ST; exit 1; }
echo "$(date '+%F %T') [submission] $NAME DONE -> $SUB/bundle (upload clean/disp1_submission.hdf5 + robust/disp1_robustness.hdf5)" | tee -a $ST
tail -n 4 $LOG/bundle.log
