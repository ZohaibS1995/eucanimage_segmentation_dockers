#!/usr/bin/env python3

import io
import json
import os
import fnmatch
from argparse import ArgumentParser


def main(args):
    # input parameters
    metrics_data = args.metrics_data
    validation_data = args.validation_data
    challenges = args.challenges_ids
    outdir = args.outdir
    consolidated_result = args.consolidated_result
    manifest_data = os.path.join(outdir, "Manifest.json")

    # Nextflow passes list, python reads string. We need python lists
    metrics_data = [m.strip('[').strip(']').strip(',') for m in metrics_data]
    validation_data = [v.strip('[').strip(']').strip(',') for v in validation_data]

    # This is the final consolidated output
    data_model_file = []

    # get the output files from previous steps and concatenate
    # 1. from validation ("validated_participant_data")
    for v in validation_data:
        data_model_file = join_json_files(v, data_model_file, "*.json")

    # 2. proceed with objects from Manifest...
    data_model_file = join_json_files(manifest_data, data_model_file, "*.json")

    # ...and 3. from metrics ("assessment_out")
    for m in metrics_data:
        data_model_file = join_json_files(m, data_model_file, "*.json")

    # 4. from consolidation part 1 (manage_assessment_data.py), "sample_out/results/challenge/challenge.json"
    # we have to do that for all challenges in the list
    for challenge in challenges:
        challenge = challenge.replace('.', '_')
        c_aggregation_data = os.path.join(outdir, challenge)
        data_model_file = join_json_files(c_aggregation_data, data_model_file, "*" + challenge + "*.json")

    # write the merged data model file to json output
    with open(consolidated_result, mode='w', encoding="utf-8") as f:
        json.dump(data_model_file, f, sort_keys=True, indent=4, separators=(',', ': '))

def join_json_files(data_directory, data_model_file, file_extension):
    '''Add contents of specified file(s) to given json file
    Input:
    data_directory: Directory containing the file(s) to be added
    data_model_file: list to be extended with json objects (dicts)
    file_extension: filename or pattern to be matched, of the file(s) to be added
    Returns:
    data_model_file: data_model_file (which is a list) from input, extended by the json objects from the input file(s)
    '''

    # add minimal datasets to data model file
    if os.path.isfile(data_directory):
        with io.open(data_directory, mode='r', encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict):
                data_model_file.append(content)
            else:
                data_model_file.extend(content)

    elif os.path.isdir(data_directory):  # if it is a directory loop over all files and search for the file with the given "extension" (=pattern, can be name)

        for subdir, dirs, files in os.walk(data_directory):
            for file in files:
                abs_result_file = os.path.join(subdir, file)
                if fnmatch.fnmatch(file, file_extension) and os.path.isfile(abs_result_file):
                    with io.open(abs_result_file, mode='r', encoding="utf-8") as f:
                        content = json.load(f)
                        if isinstance(content, dict):
                            data_model_file.append(content)
                        else:
                            data_model_file.extend(content)

    return data_model_file



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-v", "--validation_data", nargs="+", help="path to validated_result.json", required=True)
    parser.add_argument("-m", "--metrics_data", nargs="+", help="path to assessment_results.json", required=True)
    parser.add_argument("-c", "--challenges_ids", help="Ids of the challenges, separated by space", nargs='+', required=True)
    parser.add_argument("-a", "--outdir", help="output path where the minimal dataset JSON file will be written", required=True)
    parser.add_argument("-o", "--consolidated_result", help="Path to the consolidated result JSON file", required=True)

    args = parser.parse_args()

    main(args)

