
import os
import json
import logging
from copy import deepcopy
from enum import Enum
from argparse import ArgumentParser, RawTextHelpFormatter
from assessment_chart import assessment_chart


class Visualisations(Enum):
    """
    Visualisations supported for plotting.
    """
    BARPLOT = "bar-plot"
    TWODPLOT = "2D-plot"


def parse_arguments():
    '''
    Parser of command-line arguments.
    '''
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "-a",
        "--assessment_data",
        nargs="+",
        help="List of paths to participant assessment.json", required=True
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="output directory where the manifest and output JSON files will be written",
        required=True
    )
    parser.add_argument(
        "-e",
        "--event_id",
        help="Benchmarking event id / name",
        required=True
    )
    parser.add_argument(
        "-t",
        "--template",
        help="path to the aggregation template",
        #default=os.path.join(os.path.dirname(
        #    os.path.realpath(__file__)), "aggregation_aggregation_template.json"),
        required=True
    )
    return parser


def main(options):

    # Input parameters
    assessment_data = options.assessment_data
    outdir = options.outdir
    event = options.event_id
    aggregation_template=options.template

    #print('HERE', assessment_data, outdir)
    print('HERE', assessment_data, outdir)

    logging.info(f"Using event: {event}")

    # Nextflow passes list, python reads list of strings. We need to clean up list elements
    assessment_data = [a.strip('[').strip(']').strip(',')
                       for a in assessment_data]

    logging.info(f"Assessment data: {assessment_data}")

    # Assuring the output directory for participant does exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ########################################################
    # 1. Get assesment (=metrics) for current participant
    ########################################################

    # Load info from assessment file into dict of challenges with dicts of metric IDs
    community_id, participant_id, challenges = get_metrics_per_challenge(
        assessment_data)
    # Get a list of challenge ids
    challenges_ids = list(challenges.keys())
    # Get a list of metrics ids, as those change depending on the window sizes specified for compute_metrics.py
    metrics_ids = list(challenges[challenges_ids[0]].keys())

    logging.debug(
        f"Participant {participant_id}, challenges {challenges}, challenges_ids {challenges_ids}, metrics_ids {metrics_ids}")

    ########################################################
    # 2. Handle aggregation file(s)
    # Loop over challenges IDs, for each challenge:
    # a) start fresh with the provided template
    # b) add current participant's metrics to aggregation
    # c) write aggregation file
    # d) split up assessments into challenges dirs
    # e) store info for summary file (manifest)
    ########################################################

    # Store info for summary file
    manifest = []

    for challenge_id in challenges_ids:

        challenge_id_results = challenge_id.replace('.', '_')
        challenge_dir = os.path.join(outdir, challenge_id_results)

        if not os.path.exists(challenge_dir):
            os.makedirs(challenge_dir)

        # 2.a) Load the aggregation template file
        aggregation = load_aggregation_template(
            aggregation_template, community_id, event, challenge_id, metrics_ids)

        # if something else than the file missing went wrong
        logging.debug(f"aggregation on load: {aggregation}")

        # 2.b) Add the current participant's metrics to the aggregation for the current challenge_id
        new_aggregation = add_to_aggregation(
            aggregation, participant_id, challenges[challenge_id])

        logging.debug(f"aggregation after update: {new_aggregation}")

        # 2.c) Write aggregation in a file.
        aggregation_file = os.path.join(
            challenge_dir, challenge_id_results + ".json")
        # Create others aggregations in one file
        with open(aggregation_file, mode='w', encoding="utf-8") as f:
            json.dump(new_aggregation, f, sort_keys=True,
                      indent=4, separators=(',', ': '))

        # 2.d) Write assessments per challenge to local results dir
        # We have stored the assessment json objects for each challenge in the challenges dict
        challenge_assessments = []
        for metric, ass_json in challenges[challenge_id].items():
            challenge_assessments.append(ass_json)

        assessment_file = os.path.join(challenge_dir, participant_id + ".json")
        with open(assessment_file, mode='w', encoding="utf-8") as f:
            json.dump(challenge_assessments, f, sort_keys=True,
                      indent=4, separators=(',', ': '))

        # 2.e) store manifest object for current challenge
        # For that, get a list of participants from aggregation
        participants = []

        # Already recorded participants
        # Dirty: we're only looking at the first aggregation object, assuming the participants are the same for all objects (=plots)
        print('HERE', new_aggregation)
        for item in new_aggregation[0]["datalink"]["inline_data"]["challenge_participants"]:
            participants.append(item["participant_id"])

        logging.debug(f"participants: {participants}")

        mani_obj = {
            "id": challenge_id,
            "participants": participants,
        }

        manifest.append(mani_obj)

        # Create plots for current challenge
        for aggr_object in new_aggregation:
            # 2D-plots
            if aggr_object["datalink"]["inline_data"]["visualization"]["type"] == Visualisations.TWODPLOT.value:
                assessment_chart.print_chart(
                    challenge_dir, aggr_object, challenge_id, "RAW")
                assessment_chart.print_chart(
                    challenge_dir, aggr_object, challenge_id, "SQR")
                assessment_chart.print_chart(
                    challenge_dir, aggr_object, challenge_id, "DIAG")
            # barplots
            elif aggr_object["datalink"]["inline_data"]["visualization"]["type"] == Visualisations.BARPLOT.value:
                assessment_chart.print_barplot(
                    challenge_dir, aggr_object, challenge_id)

    # After we have updated all aggregation files for all challenges, save the summary manifest
    with open(os.path.join(outdir, "Manifest.json"), mode='w', encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True,
                  indent=4, separators=(',', ': '))


# Function definitions
def get_metrics_per_challenge(assessment_data):
    '''
    From the assessment file collect all challenges and all metrics/assessments per challenge
    Returns:
    participant ID
    dict of challenges, per challenge dict of metric IDs

    challenges = {
        "ECI_UC0_Seg": {
            "dsc_0.0": {ASSESSMENT OBJECT},
            "dsc_2.0": {ASSESSMENT OBJECT},
            "dsc_3.0": {ASSESSMENT OBJECT},
            ...
        },
        "OTHER CHALLENGE": {
            ...,
            ...,
            ...
        }
    }
    '''
    # read assessment files
    assessments = []
    for ass in assessment_data:
        with open(ass, mode='r', encoding="utf-8") as f:
            a = json.load(f)
        assessments.extend(a)

    logging.debug(f"Assessments: {assessments}")

    # initialize
    challenges = {}
    participant_id = assessments[0]["participant_id"]
    community_id = assessments[0]["community_id"]

    # go through all assessment objects
    for item in assessments:

        # make sure we're dealing with assessment objects
        assert_object_type(item, "assessment")
        # get metric
        metric_id = item["metrics"]["metric_id"]
        challenge_id = item["challenge_id"]
        # add metric assessment object to dict of its challenge
        challenges.setdefault(challenge_id, {})[metric_id] = item

        # make sure all objects belong to same participant
        if item["participant_id"] != participant_id:
            raise ValueError(
                "Something went wrong, not all metrics in the assessment file belong to the same participant.")

    return community_id, participant_id, challenges


def assert_object_type(json_obj, curr_type):
    '''
    Check OEB json object type
    Input:
    json object
    desired object type (e.g. "assessment", "aggregation", "participant")
    Returns:
    None; raises TypeError if object is not of given type
    '''
    type_field = json_obj["type"]
    if type_field != curr_type:
        raise TypeError(
            f"json object is of type {type_field}, should be {curr_type}")

#logging.info(f"Using event: {EVENT}")

def load_aggregation_template(aggregation_template, community_id, event, challenge_id, metrics_ids):
    '''
    Load the aggregation template from the provided json file and set _id and challenge_id; create objects for all window sizes
    '''

    with open(aggregation_template, mode='r', encoding="utf-8") as t:
        template = json.load(t)

    aggregation = []

    # Create aggregation objects for all plots specified in the template, and for all metrics detected in the assessment. Fill objects with ID and challenge ID; data will be appended later
    # Prefix for aggregation object ids
    base_id = f"{community_id}:{event}_{challenge_id}_agg:"

    # For all different plots
    for item in template:
        viz = item["datalink"]["inline_data"]["visualization"]

        # 2D-plot
        if viz["type"] == Visualisations.TWODPLOT.value:
            x = viz["x_axis"]
            y = viz["y_axis"]

            # one plot (=aggregation object) per window size
            for x_win in [m for m in metrics_ids if x in m]:
                # new aggregation object
                win_item = deepcopy(item)

                win_size = x_win.split(":")[-1]
                y_win = y.split(":")[0]  # get the base part of y without window size
                win_metrics = f"{x_win}_vs_{y_win}"
                win_item["_id"] = base_id + win_metrics
                win_item["challenges_ids"] = [challenge_id]
                win_item["datalink"]["inline_data"]["visualization"]["x_axis"] = x_win
                win_item["datalink"]["inline_data"]["visualization"]["y_axis"] = y_win
                aggregation.append(win_item)

        # bar-plot
        elif viz["type"] == Visualisations.BARPLOT.value:
            y = viz["metric"]

            # one plot (=aggregation object) per window size
            for y_win in [m for m in metrics_ids if y in m]:
                # new aggregation object
                win_item = deepcopy(item)

                win_metrics = f"{y_win}"
                win_item["_id"] = base_id + win_metrics
                win_item["challenges_ids"] = [challenge_id]
                win_item["datalink"]["inline_data"]["visualization"]["metric"] = y_win
                aggregation.append(win_item)

        # someting wrong
        else:
            raise KeyError("Unknown plot type")

    return aggregation


def add_to_aggregation(aggregation, participant_id, challenge):
    '''
    Add the metrics for the current challenge to the challenge's aggregation file. Aggregation file can have more than one aggregation object, one per plot type.
    '''
    for item in aggregation:
        assert_object_type(item, "aggregation")

        # get the current visualization object
        plot = item["datalink"]["inline_data"]["visualization"]

        # Depending on the type of plot we'll need to create different participant objects
        participant = {}
        participant["participant_id"] = participant_id
        skip_participant = False
        if plot["type"] == Visualisations.TWODPLOT.value:
            try:
                participant["metric_x"] = challenge[plot["x_axis"]
                                                    ]["metrics"]["value"]
                participant["metric_y"] = challenge[plot["y_axis"]
                                                    ]["metrics"]["value"]
            except Exception as e:
                logging.exception(str(e))
                skip_participant = True

        elif plot["type"] == Visualisations.BARPLOT.value:
            try:
                participant["metric_value"] = challenge[plot["metric"]
                                                        ]["metrics"]["value"]
            except KeyError as e:
                logging.exception(
                    f"The assessment file does not contain data for metric {plot['metric']}.")
                raise e

        # Append current participant to the list of participant objects
        if not skip_participant: 
            item["datalink"]["inline_data"]["challenge_participants"].append(
                participant)

    return aggregation


if __name__ == '__main__':
    try:
        # parse the command-line arguments
        options = parse_arguments().parse_args()

        # set up logging during the execution
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
        # execute the body of the script
        logging.info("Starting script")
        main(options)
        logging.info("Finished script successfully.")

    except Exception as e:
        logging.exception(str(e))
        raise e
