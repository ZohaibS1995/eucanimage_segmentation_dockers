#!/bin/bash -ue
python3 /app/compute_metrics.py -i brain.nii.tar.gz -c ECI_UC0_Seg -e Brain_Cancer_Diagnosis -g goldstandard_dir -p tool1 -com EuCanImage -o "assessment_results.json"
