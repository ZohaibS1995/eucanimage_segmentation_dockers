# Brain Cancer Diagnosis benchmark specification

## Synopsis

Example benchmark to test median dice similarity coefficient (DSC) scores in NIfTI brain images compared with ground truth.

The metrics are computed for five types of datasets:

- IXI015-HH-1258-T1
- IXI014-HH-1236-T1
- IXI013-HH-1212-T1
- IXI012-HH-1211-T1
- IXI002-Guys-0828-T1

### Input data:

Comparison of image segmentation in brain NIfTI files

### Metrics

Based on the input data the following metrics are computed:

1. dsc_0.0
2. dsc_2.0
3. dsc_3.0
4. dsc_4.0
5. dsc_5.0
6. dsc_7.0
7. dsc_8.0
8. dsc_10.0
9. dsc_11.0
10. dsc_12.0

### OpenEBench challenges

The metrics are visualised using 2D scatter plots and barplots, as described in _Plots_ section, which are then used for ranking the participating tools.
The plots for a given input dataset belong to one benchmarking challenge as understood in OpenEBench schema.
Not all plots have to be prepared for each dataset, as described in _Plots_ section.
In this example, only one challenge has been described. However, if more than one challenge is included all challenges belong to the same benchmarking event.

## Inputs


| # | Format | Link                                                                                      | Example data |
| - | ------ | ----------------------------------------------------------------------------------------- | ------------ |
| 1 | NIfTI  | [Wikipedia][wiki-NIfTI] | [Link][in1]  |
| 2 | NIfTI  | [Wikipedia][wiki-NIfTI] | [Link][in2]  |

## Plots

The results of this benchmark will be visualised in OpenEBench using the following plots:

### 1. **Bar plot visualizing **dsc_10.0** of image segementation.

**Plot type**: bar plot

**Metric**: dsc_10.0

**Ranking**: The best performing tool is the one with the highest dsc_10.0.

### 2. **Bar plot visualizing **dsc_8.0** of image segementation.

**Plot type**: bar plot

**Metric**: dsc_10.0

**Ranking**: The best performing tool is the one with the highest dsc_8.0.

### 3. **Bar plot visualizing **dsc_3.0** of image segementation.

**Plot type**: bar plot

**Metric**: dsc_3.0

**Ranking**: The best performing tool is the one with the highest dsc_3.0.

### 4. **2D scatter plots** visualizing **dsc_0.0 and dsc_2.0** of image segmentation.

**Plot type**: 2D scatter plot

**Metric X**: dsc_0.0
**Metric Y**: dsc_2.0

**Ranking**: The best performing tool is the one with the highest dsc_2.0 combined with lowest dsc_0.0 (top left part of the plot) and the worst performing tool is the one with the lowest dsc_2.0 combined with highest dsc_0.0 (bottom right part of the plot). The plot should be divided into diagonal quartiles based on the distance from optimal performance.

### 5. **2D scatter plot** visualizing **dsc_4.0 and dsc_5.0** of image segmentation.

**Plot type**: 2D scatter plot

**Metric X**: dsc_4.0
**Metric Y**: dsc_5.0

**Ranking**: The best performing tool is the one with the highest dsc_4.0 combined with lowest dsc_5.0 (bottom right part of the plot) and the worst performing tool is the one with the lowest dsc_4.0 combined with highest dsc_5.0 (top left part of the plot). The plot should be divided into diagonal quartiles based on the distance from optimal performance.

## Outputs

Calculated metrics are saved in JSON file adhering to OpenEBench schema.
Assessment output is generated for each tool separately and contains values of calculated metrics for a given input dataset.
Aggregation output contains summarized metrics values from all benchmarked tools for all plots of the given challenge.
Consolidation output combines assessment data for the participant currently benchmarked with aggregation of previously benchmarked participants and will be uploaded to the OEB DB.
Results example contains a minimal dataset for one benchmarking event collecting all participants, and challenges; and composed of participant, aggregation, and assessment datasets, among other required data. Example for one existing community is available.


| # | Format | Link                       | Example data            | Description               |
| :- | :----- | :------------------------- | :---------------------- | :------------------------ |
| 1 | JSON   | [Specification][spec-json] | [Link][assessment_out]  | Assessment output JSON    |
| 1 | JSON   | [Specification][spec-json] | [Link][assessment_G_STD]  | Assessment output JSON    |
| 2 | JSON   | [Specification][spec-json] | [Link][aggregation_out]  | Aggregation output JSON    |
| 2 | JSON   | [Specification][spec-json] | [Link][aggregation_ECtest]  | Aggregation output JSON    |
| 3 | JSON   | [Specification][spec-json] | [Link][consolidation_out]  | Consolidation output JSON    |
| 4 | JSON   | [Specification][spec-json] | [Link][results_example]  | Results example JSON    |

### Additional info

#### Output 1

The OpenEBench assessment file contains the following attributes:

- **\_id** - follows the format: [COMMUNITY_ID]:[EVENT_ID]\_[CHALLENGE_ID]\_[PARTICIPANT_ID]\:[METRIC_ID] 
  All default separators, except by the separator among[PARTICIPANT_ID]\:[METRIC_ID]; which has been customized from `_` to `:` to be more readable; but they can be customized according to any Community's requests. 
- **challenge_id** - challenge name
- **participant_id** - benchmarked tool
- **community_id** - community name
- **metrics**
  - **value** - metric value
  - **metric_id** - metric name; metric names used in this benchmark are specified in the table below
- **type** - type of dataset (_aka_ assessment)

#### Output 2

The OpenEBench aggregation file contains information required to produce the plots. 
The `_id` attribute follows this convention:

- **\_id** - follows the format: [COMMUNITY_ID]:[EVENT_ID]\_[CHALLENGE_ID]\_[PARTICIPANT_ID]\:[METRIC_ID], for a single metrics visualization, and
COMMUNITY_ID]:[EVENT_ID]\_[CHALLENGE_ID]\_[PARTICIPANT_ID]\:[METRIC_ID1]\_vs\_[METRIC_ID2], for a paired metrics visualization
  All default separators, except by the separator among[PARTICIPANT_ID]\:[METRIC_ID]; which has been customized from `_` to `:` and \_vs\_ among metrics, to be more readable; but they can be customized according to any Community's requests. 
  You can explore all OEB data model naming conventions [here](https://github.com/inab/OEB_level2_data_migration/blob/master/NAMING-CONVENTIONS.md).

. 

#### Output 3

The OpenEBench consolidation file contains all the information about the new benchmarking run and specifies visualization types (2D scatter plot, barplot), descriptions of metrics used for X and Y axis and (X,Y) value pairs for each challenge and challenge participants. A basic consolidation file is generated during the pipeline by using the `minimal_aggregation_template.json` file. However, as there is no data from other participants, the consolidation file generated cannot show a complete consolidation in this workflow example.

[//]: #
[in1]: ./example_files/brain_IXI014-HH-1236-T1.nii.gz
[in2]: ./example_files/brain_IXI015-HH-1258-T1.nii.gz
[assessment_out]: ./example_files/assessment_out.json
[assessment_G_STD]: ./example_files/assessment_G_STD.json
[aggregation_out]: ./example_files/aggregation_out.json
[aggregation_ECtest]: ./example_files/aggregation_ECtest.json
[consolidation_out]: ./example_files/consolidation_out.json
[results_example]: ./example_files/results_example.json
[wiki-NIfTI]: https://en.wikipedia.org/wiki/Neuroimaging_Informatics_Technology_Initiative