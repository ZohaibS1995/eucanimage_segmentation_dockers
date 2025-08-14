#!/usr/bin/env nextflow
nextflow.enable.dsl=2

if (params.help) {
	
	log.info"""
	    =====================================================
	    BENCHMARKING PIPELINE IN OPENEBENCH
	    Author(s): Anna Redondo, The OpenEBench Team 
		Barcelona Supercomputing Center, Barcelona, Spain
		https://openebench.bsc.es/team
	    =====================================================
	    Usage:
	    Run the pipeline with default parameters read from nextflow.config:
	    nextflow run main.nf -profile docker
	    Run with user parameters:
	    nextflow run main.nf -profile docker --input {wf.prediction.file} --community_id {community.id} --event_id {event.id} --participant_id {participant.id} --goldstandard_dir {goldstandard.dir} --challenges_ids {analyzed.challenges} --template {template.path} --outdir {outdir.dir} 
		nextflow run main.nf -profile docker -c {parameters_file.config}
		Mandatory arguments:
	        --input                 File name with prediction results for the event (in nii.gz.tar format)
	        --community_id          Name or OEB permanent ID for the benchmarking community
			--challenges_ids        Challenges ids selected by the user, separated by spaces
	        --participant_id        Name or OEB permanent ID of the tool/model used for prediction
		    --goldstandard_dir      Dir that contains gold standard/ground truth files used to calculate the metrics for all challenges
			--validation_result     Output path of the JSON file where the validated participant file is written
	        --assessment_results    Output path of the JSON files where where the set of assessment datasets are written (these correspond to minimal datasets compatible with the OEB Data Model)
			--consolidated_result   Output path of the JSON file (consolidated_result) where all the datasets generated during the workflow are merged
	    	--outdir                The output directory where resulting datsets will be saved
			--results_dir			The output directory where the results will be saved
	        --statsdir              The output directory with nextflow statistics

		Other options:
			--event_id				Name or OEB permanent ID for the benchmarking event 
			--template    			Path to the JSON template file with the minimal data for the aggregation step to obtain the minimal benchmark data 
		//  --public_ref_dir     	Dir that can contain public reference file(s) used to validate the input parameters
	    //  --otherdir              Output directory where custom results can be saved (no directory inside)
			 
		Flags:
	        --help                  Display this message
	    """.stripIndent()

	exit 1
} else {

	log.info """\

	    ==============================================
	    BENCHMARKING PIPELINE IN OPENEBENCH
	    ==============================================
	        Input file: ${params.input_file}
	        Benchmarking community = ${params.community_id}
			Challenges ids: ${params.challenges_ids}
			Participant id : ${params.participant_id}
			Gold standard dataset DIR: ${params.goldstandard_dir}
			Validation result JSON file: ${params.validation_result}
            Assessment results JSON files: ${params.assessment_results}
			Consolidation result JSON file: ${params.consolidated_result}
	        Benchmark datasets DIR: ${params.outdir}
	        Nextflow statistics DIR: ${params.statsdir}
            Benchmarking event: ${params.event_id}
		    Minimal aggregation data template JSON file: ${params.template_path}
			// Public reference data DIR: ${params.public_ref_dir}
	        // Community-specific results DIR: ${params.otherdir}

		""".stripIndent()

}

// Input

input_file = Channel.fromPath(params.input_file, type: 'file')
participant_id = params.participant_id.replaceAll("\\s","_")
goldstandard_dir = Channel.fromPath(params.goldstandard_dir, type: 'dir' ) 
challenges_ids = params.challenges_ids
community_id = params.community_id
event_id = params.event_id
template_path = Channel.fromPath(params.template_path, type: 'file')
//public_ref_dir = Channel.fromPath(params.public_ref_dir, type: 'dir' ) 

// Output

outdir = file(params.outdir, type: 'dir')
validation_result = file(params.validation_result)
assessment_results = file(params.assessment_results)
consolidated_result = file(params.consolidated_result)
//otherdir = file(params.otherdir, type: 'dir')

// Process definitions

process validation {

	// validExitStatus 0,1
	tag "Validating input file format"
	
	publishDir outdir,
	mode: 'copy',
	overwrite: false,
    saveAs: { filename -> "validated_result.json" }

    // Publish validation_result copy in OEB VRE only
	publishDir validation_result.parent,
	mode: 'copy',
	overwrite: false,
    saveAs: { filename -> 
		def DEFAULT_VALIDATION_RESULT = "${params.outdir}/validated_result.json"
        // Convert both paths to absolute paths for comparison
		def fullValidationResultPath = validation_result.toString()
        def fullDefaultValidationResultPath = DEFAULT_VALIDATION_RESULT.toString()
        return fullValidationResultPath == fullDefaultValidationResultPath ? null : validation_result.name 
    }

	input:
	//path default_validation_result
	path input_file
	val challenges_ids
	val participant_id
	val community_id
	val event_id
	// path public_ref_dir
	path goldstandard_dir
	
	output:
    path "validated_result.json", emit: validation_file
	val task.exitStatus, emit: validation_status
			
	"""
	python3 /app/validation.py -i $input_file -com $community_id -c $challenges_ids -e $event_id -p $participant_id -g $goldstandard_dir 
	"""

}

default_assessment_filename = "assessment_results.json"

process compute_metrics {

	tag "Computing benchmark metrics for submitted data"
	
	publishDir outdir,
	mode: 'copy',
	overwrite: false,
    saveAs: { filename -> default_assessment_filename }

    // Publish assessment_results copy in OEB VRE only
	publishDir assessment_results.parent,
	mode: 'copy',
	overwrite: false,
	saveAs: { filename -> 
		def DEFAULT_ASSESSMENT_RESULTS = "${params.outdir}/${default_assessment_filename}"
		// Convert both paths to absolute paths for comparison
        def fullAssessmentResultsPath = assessment_results.toString()
        def fullDefaultAssessmentResultsPath = DEFAULT_ASSESSMENT_RESULTS.toString()
        return fullAssessmentResultsPath == fullDefaultAssessmentResultsPath ? null : assessment_results.name 
    }

	input:
	val validation_status
	path input_file
	val challenges_ids
	path goldstandard_dir
	val participant_id
	val community_id
	val event_id

	output:
    path "${default_assessment_filename}", emit: ass_json
	
	when:
	validation_status == 0

	"""
	python3 /app/compute_metrics.py -i $input_file -c $challenges_ids -e $event_id -g $goldstandard_dir -p $participant_id -com $community_id -o "${default_assessment_filename}"
	
	"""
}


default_consolidation_filename = "consolidated_result.json"

process benchmark_consolidation {

	tag "Performing benchmark assessment and building plots"

	publishDir outdir, 
	mode: 'copy',
	overwrite: false,
	saveAs: { filename -> default_consolidation_filename }

    // Publish consolidated_result copy in OEB VRE only
	publishDir consolidated_result.parent,
	mode: 'copy',
	overwrite: false,
	saveAs: { filename -> 
		def DEFAULT_CONSOLIDATED_RESULT = "${params.outdir}/${default_consolidation_filename}"
		// Convert both paths to absolute paths for comparison
        def fullConsolidatedResultPath = consolidated_result.toString()
        def fullDefaultConsolidatedResultPath = DEFAULT_CONSOLIDATED_RESULT.toString()
        return fullConsolidatedResultPath == fullDefaultConsolidatedResultPath ? null : consolidated_result.name
    }

	input:	

	path ass_json
	val event_id
	path outdir
	path template_path
	path validation_file
	val challenges_ids
	
	output:
	path "${default_consolidation_filename}", emit: consolidated_result
	
	"""
	python /app/aggregation.py -a $ass_json -e $event_id -o $outdir -t $template_path
	python /app/merge_data_model_files.py -m $ass_json -v $validation_file -c $challenges_ids -a $outdir -o "${default_consolidation_filename}"
	"""

}

// Workflow

workflow {
    // Collect inputs and parameters
    validation(
        input_file,
        challenges_ids,
        participant_id,
        community_id,
		event_id,
        // public_ref_dir,
        goldstandard_dir
    )
    validations = validation.out.validation_file.collect()

	compute_metrics(
		validation.out.validation_status,
		input_file,
		challenges_ids,
		goldstandard_dir,
		participant_id,
		community_id,
		event_id
	)
	assessments = compute_metrics.out.ass_json.collect()

	benchmark_consolidation(
		assessments,
		event_id,
		outdir,
		template_path,
		validation_result,
		challenges_ids
	)
}

workflow.onComplete { 
	println ( workflow.success ? "Done!" : "Oops .. something went wrong" )
}
