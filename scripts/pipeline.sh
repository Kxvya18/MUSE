#!/bin/bash
# End-to-end journal pipeline (runs unattended). Progress: logs_pipeline.txt
cd "$(dirname "$0")/.."
source .venv/bin/activate
L=logs_pipeline.txt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
pr(){ local b; b=$(mktemp); printf "%s\n" "$3" > "$b"; scripts/pr_results.sh "$1" "$2" "$b" >> $L 2>&1 || log "PR step failed: $1"; rm -f "$b"; }
MAIN=MUSE,GCN-zero,GAT-zero,APPNP-zero,MMGL-style,TMC,ECD,EGNN,HeMIS
MAINB=GCN-zero,GAT-zero,APPNP-zero,MMGL-style,TMC,ECD,EGNN,HeMIS
ABL=MUSE-noProp,MUSE-noAux,MUSE-selfEvidence,MUSE-induced,MUSE-noGate,MUSE-Dempster,MUSE-zeroimpute,MUSE-specVar,MUSE-noVar,MUSE-MLP

graph_lane(){
  log "graph: tuning Cora (MUSE + remaining baselines, small grid)"
  python run_graph.py tune --datasets Cora --models MUSE --grid small > logs_g_tune1.txt 2>&1 &
  python run_graph.py tune --datasets Cora --models TMC,HeMIS,MMGL-style,ECD,EGNN --grid small > logs_g_tune2.txt 2>&1
  wait
  log "graph: tuning CiteSeer (small grid)"
  python run_graph.py tune --datasets CiteSeer --models MUSE --grid small > logs_g_tune3.txt 2>&1 &
  python run_graph.py tune --datasets CiteSeer --models $MAINB --grid small > logs_g_tune4.txt 2>&1
  wait
  log "graph: tuning Photo (photo grid, seed 0)"
  python run_graph.py tune --datasets Photo --models MUSE --grid photo --seeds 0 > logs_g_tune5.txt 2>&1 &
  python run_graph.py tune --datasets Photo --models $MAINB --grid photo --seeds 0 > logs_g_tune6.txt 2>&1
  wait
  python run_graph.py select > results/configs_selected.txt 2>&1
  log "graph: selected configs"; cat results/configs_selected.txt >> $L
  pr results/graph-tuning "Graph benchmarks: validation tuning results and selected configs" \
     "Validation-only grid search on Cora/CiteSeer/Photo; best config per (dataset, model) in results/configs.json. Test set not touched."
  log "graph: final runs (5 seeds)"
  python run_graph.py final --datasets Cora,CiteSeer,Photo --models $MAIN > logs_g_final1.txt 2>&1 &
  python run_graph.py final --datasets Cora,CiteSeer --models $ABL > logs_g_final2.txt 2>&1
  wait
  log "graph: order-dependence analysis"
  python analysis_order.py > logs_order.txt 2>&1
  pr results/graph-final "Graph benchmarks: final 5-seed test results, ablations, order-dependence" \
     "Final test results (seeds 100-104) for all methods and MUSE ablations on Cora/CiteSeer/Photo; order-dependence of fusion rules on trained evidence."
  log "graph lane done"
}

mri_lane(){
  log "mri: building multi-scale supervoxel graphs"
  python preprocess_brats.py --cube 4 --workers 4 > logs_preprocess_ms.txt 2>&1
  log "mri: MUSE and GAT-zero (seed 100) for qualitative panels"
  python run_brats.py final --models MUSE,GAT-zero --seeds 100 > logs_m_final0.txt 2>&1
  python figures/make_mri_panels.py --baseline GAT-zero --n 4 >> $L 2>&1 || log "panels failed"
  pr results/mri-panels "MRI: qualitative panels (MUSE vs GAT-zero, seed 100)" \
     "Real FLAIR/T1/T1gd/T2 slices with ground truth, MUSE segmentation and uncertainty, GAT-zero segmentation and per-sequence contributions for 4 missing-sequence patterns on held-out patients (multi-scale features)."
  log "mri: final runs, all methods, seeds 100-102"
  python run_brats.py final --models $MAIN --seeds 100,101,102 > logs_m_final.txt 2>&1
  log "mri: ablations, seed 100"
  python run_brats.py final --models MUSE-noProp,MUSE-noAux,MUSE-noGate,MUSE-Dempster,MUSE-zeroimpute,MUSE-specVar --seeds 100 > logs_m_abl.txt 2>&1
  pr results/mri-final "MRI: final results (3 seeds) and ablations" \
     "Brain-tumour MRI (MSD Task01, 98 held-out patients): Dice WT/TC/ET for all 15 missing-sequence patterns, calibration, error detection, boundary uncertainty; MUSE ablations (seed 100)."
  log "mri lane done"
}

log "pipeline (re)start"
graph_lane &
mri_lane &
wait
log "aggregating tables and figures"
python aggregate.py >> $L 2>&1
python figures/make_plots.py >> $L 2>&1
best=$(python -c "
import json;d=json.load(open('results/tables/mri_per_pattern.json'));import numpy as np
b=[m for m in d if m!='MUSE' and not m.startswith('MUSE-')]
print(max(b,key=lambda m:np.mean([np.mean(list(v.values())) for v in d[m].values()])))" 2>/dev/null || echo GAT-zero)
python figures/make_mri_panels.py --baseline "$best" --n 4 >> $L 2>&1 || log "final panels failed"
pr results/tables-figures "Journal tables and figures from final results" \
   "LaTeX/Markdown tables with paired significance tests (results/tables) and result figures (figures/results, figures/mri)."
log "PIPELINE COMPLETE"
