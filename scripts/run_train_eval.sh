#!/usr/bin/env bash
# Usage: scripts/run_train_eval.sh <gpu> <config> <outdir name>
# Full local training schedule + frozen robust20 evaluation into $W/outputs/track2/final_runs/<name>/ (refuses to overwrite an existing final.pt).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."; W=${ROCO_TRACK2_WORK:-$PWD}
G=$1; C=$2; D=$W/outputs/track2/final_runs/$3; ST=$W/outputs/track2/pipeline_status.txt
mkdir -p $D
{ echo "date: $(date '+%F %T')"; echo "gpu: $G"; echo "config: $C"
  echo "train: CUDA_VISIBLE_DEVICES=$G python3 -u scripts/train_track2.py --config $C --out $D/train"
  echo "eval : CUDA_VISIBLE_DEVICES=$G python3 -u scripts/eval_suite.py --suite robust20 --backbone croco --checkpoint $D/train/final.pt --tag robust20 --out $D/eval"
  echo "git: unavailable on this host; source hashes are in train/contract.json"; } > $D/COMMANDS.txt
echo "$(date '+%F %T') [train-eval] GPU$G start $C -> $D" | tee -a $ST
if [ ! -f $D/train/final.pt ]; then
  [ -d $D/train ] && mv $D/train $D/train_incomplete_$(date +%s)
  CUDA_VISIBLE_DEVICES=$G python3 -u scripts/train_track2.py --config $C --out $D/train > $D/train.log 2>&1 || { echo "$(date '+%F %T') [train-eval] TRAIN FAILED $C" | tee -a $ST; exit 1; }
fi
CUDA_VISIBLE_DEVICES=$G python3 -u scripts/eval_suite.py --suite robust20 --backbone croco --checkpoint $D/train/final.pt --tag robust20 --out $D/eval > $D/eval.log 2>&1 || { echo "$(date '+%F %T') [train-eval] EVAL FAILED $C" | tee -a $ST; exit 1; }
echo "$(date '+%F %T') [train-eval] DONE $C" | tee -a $ST
