#!/bin/bash -ue
python3 /app/validation.py -i brain.nii.tar.gz -com EuCanImage -c ECI_UC0_Seg -e Brain_Cancer_Diagnosis -p tool1 -g goldstandard_dir
