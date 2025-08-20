#!/bin/bash -ue
python /app/aggregation.py -a assessment_results.json -e ECI_UC7_Seg -o output_data -t minimal_aggregation_template.json
python /app/merge_data_model_files.py -m assessment_results.json -v validated_result.json -c ECI_UC7_Seg -a output_data -o "consolidated_result.json"
