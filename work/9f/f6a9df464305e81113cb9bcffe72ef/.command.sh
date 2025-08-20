#!/bin/bash -ue
python3 /app/compute_metrics.py -i participant_dir -c ECI_UC7_Seg -e ECI_UC7_Seg -g goldstandard_dir -p tool1 -com EuCanImage -o "assessment_results.json"
