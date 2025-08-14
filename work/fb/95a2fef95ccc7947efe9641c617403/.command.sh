#!/bin/bash -ue
python3 /app/validation.py -i segmentations.tar.gz -com EuCanImage -c ECI_UC0_Seg -e Brain_Cancer_Diagnosis -p tool1 -g goldstandard_dir
