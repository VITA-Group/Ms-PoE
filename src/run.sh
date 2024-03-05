nohup bash src/ms_poe.sh 1 0 > log_ours_1.out 2>&1 &
nohup bash src/ms_poe.sh 3 1 > log_ours_3.out 2>&1 &

nohup bash src/ms_poe.sh 5 2 > log_ours_5.out 2>&1 &
nohup bash src/ms_poe.sh 7 3 > log_ours_7.out 2>&1 &
nohup bash src/ms_poe.sh 10 4 > log_ours_10.out 2>&1 &


# nohup bash src/baseline.sh 1 0 > log_baseline_1.out 2>&1 &
# nohup bash src/baseline.sh 3 1 > log_baseline_3.out 2>&1 &
# nohup bash src/baseline.sh 5 2 > log_baseline_5.out 2>&1 &
# nohup bash src/baseline.sh 7 3 > log_baseline_7.out 2>&1 &
# nohup bash src/baseline.sh 10 4 > log_baseline_10.out 2>&1 &


