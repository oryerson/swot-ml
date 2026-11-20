# Settings shared by every scoring and figure step. Usage: source 07_evaluation/c_scoring/env.sh
#
# IP_RUNS    the two trained seeds; scorers average their predictions per node and day
# IP_HIND    prediction tag of the 2010-2020 window
# IP_SERIES  series file of the SWOT Period scorer (built by ip_series.py)
: "${SWOTGAT_WORK:?run 'source config/paths.env' first}"
: "${SWOTGAT_RUN_S1:?set SWOTGAT_RUN_S1 and SWOTGAT_RUN_S2 in config/paths.env}"
export IP_RUNS=$SWOTGAT_RUN_S1,$SWOTGAT_RUN_S2
export IP_STEPS=$SWOTGAT_RUN_S1:step_050000,$SWOTGAT_RUN_S2:step_050000
export IP_HIND=h10
export IP_SERIES=$SWOTGAT_WORK/CONUS_storage_fork/eval/c5_paper/fig_c5/series_ip.npz
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore MPLBACKEND=Agg
