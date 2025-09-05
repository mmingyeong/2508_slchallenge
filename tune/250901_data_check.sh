#!/bin/bash
#PBS -N data_check
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o data_check.pbs.out

source ~/.bashrc
conda activate py312

cd /caefs/user/mmingyeong/2508_slchallence/tune

python data_sanity_and_split.py \
  --slsim_lenses      /caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses \
  --slsim_nonlenses   /caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses \
  --hsc_lenses        /caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses \
  --hsc_nonlenses     /caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses \
  --scan_fraction 1 \
  --num_workers 8 \
  --out_dir ./_data_check
