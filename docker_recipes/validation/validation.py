#!/usr/bin/env python3

from __future__ import division, print_function
import os
import sys
import tarfile
import json
from argparse import ArgumentParser

import nibabel as nib
import JSON_templates


# -----------------------------------------------------------------------------#
# CLI arguments
# -----------------------------------------------------------------------------#
parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True,
                    help="Tar archive containing the participant predictions.")
parser.add_argument("-com", "--community_id", required=True,
                    help="OEB community id or label, e.g. 'EuCanImage'.")
parser.add_argument("-c", "--challenges_ids", nargs='+', required=True,
                    help="Challenge name(s) chosen by the user.")
parser.add_argument("-p", "--participant_id", required=True,
                    help="Tool / model id (participant).")
parser.add_argument("-e", "--event_id", required=True,
                    help="Benchmarking event id or name.")
parser.add_argument("-g", "--goldstandard_dir", required=True,
                    help="Directory containing gt_*.nii.gz files.")
args = parser.parse_args()


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def extract_tarfile(tar_path, extract_to):
    """Extract a .tar.gz archive and return the list of extracted file paths."""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
        return [os.path.join(extract_to, m.name)
                for m in tar.getmembers() if m.isfile()]


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main(cfg):
    # ---------------------------------------------------------------------#
    # 1.  Untar participant submission
    # ---------------------------------------------------------------------#
    untar_dir = os.path.join(os.path.dirname(cfg.input), "participant_files")
    os.makedirs(untar_dir, exist_ok=True)

    if not any(os.scandir(untar_dir)):
        seg_archive_files = extract_tarfile(cfg.input, untar_dir)
    else:
        print(f"INFO: Directory {untar_dir} already exists, skipping extraction.")
        seg_archive_files = [os.path.join(untar_dir, f) for f in os.listdir(untar_dir)]

    print(f"INFO: Extracted files: {seg_archive_files}")

    # ---------------------------------------------------------------------#
    # 2.  Check filename convention (seg_*.nii.gz)
    # ---------------------------------------------------------------------#
    seg_filenames = sorted([f for f in os.listdir(untar_dir)
                            if f.startswith("seg_") and f.endswith(".nii.gz")])

    if not seg_filenames:
        sys.exit("ERROR: No files matching pattern 'seg_*.nii.gz' were found.")

    # report anything that does *not* match the convention
    non_conforming = [f for f in os.listdir(untar_dir)
                      if not (f.startswith("seg_") and f.endswith(".nii.gz"))]
    if non_conforming:
        print(f"WARNING: {len(non_conforming)} file(s) do not match the "
              f"'seg_*.nii.gz' convention and will be ignored: {non_conforming}")

    seg_files = [os.path.join(untar_dir, f) for f in seg_filenames]

    # ---------------------------------------------------------------------#
    # 3.  Gather ground-truth list
    # ---------------------------------------------------------------------#
    gt_files = sorted([os.path.join(cfg.goldstandard_dir, f)
                       for f in os.listdir(cfg.goldstandard_dir)
                       if f.startswith("gt_") and f.endswith(".nii.gz")])

    print(f"INFO: Found GT files: {gt_files}")

    # sanity-check: same number of prediction & GT files?
    if len(seg_files) != len(gt_files):
        sys.exit(f"ERROR: #prediction files ({len(seg_files)}) "
                 f"≠ #ground-truth files ({len(gt_files)}).")

    # ---------------------------------------------------------------------#
    # 4.  Per-file validation
    # ---------------------------------------------------------------------#
    for seg_file, gt_file in zip(seg_files, gt_files):
        try:
            print(f"INFO: Loading seg file: {seg_file}")
            print(f"INFO: Loading GT  file: {gt_file}")

            seg_img = nib.load(seg_file)
            gt_img = nib.load(gt_file)

            if seg_img.header.get_zooms() != gt_img.header.get_zooms():
                sys.exit(f"ERROR: Spacing mismatch between {seg_file} and {gt_file}.")

            if seg_img.shape != gt_img.shape:
                sys.exit(f"ERROR: Dimension mismatch between {seg_file} and {gt_file}.")

        except FileNotFoundError as err:
            sys.exit(f"ERROR: {err}")
        except nib.filebasedimages.ImageFileError as err:
            sys.exit(f"ERROR: Invalid NIfTI file: {err}")
        except Exception as err:
            sys.exit(f"ERROR: Unexpected problem while validating: {err}")

    # ---------------------------------------------------------------------#
    # 5.  Emit validation JSON
    # ---------------------------------------------------------------------#
    output_filename = "validated_result.json"
    data_id = f"{cfg.community_id}:{cfg.event_id}_{cfg.participant_id}"
    validated = True

    validation_json = JSON_templates.write_participant_dataset(
        data_id,
        cfg.community_id,
        cfg.challenges_ids,
        cfg.participant_id,
        validated
    )

    with open(output_filename, "w") as fp:
        json.dump(validation_json, fp, indent=4, sort_keys=True, separators=(",", ": "))

    print(f"INFO: Validation succeeded. JSON written to {output_filename}")
    sys.exit(0 if validated else 1)


if __name__ == "__main__":
    main(args)
