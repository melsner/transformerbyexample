This is the code for the exemplar-based inflector described in Elsner 2021 "What transfers in morphological inflection" and Elsner and Court 2022 "OSU at SigMorphon 2022: Analogical Inflection With Rule Features".

To create a background model for a dataset, use (e.g.):

python -u script/universalbackground.py --run <NAME> --data ../2022InflectionST/part1-byfam --s2s_max_num_epochs=100 --s2s_patience=30 --extra_features=all

This will create a set of synthetic examples, plus features for all the language names and cells, and for each character in the dataset. (No other language-specific information is used in the background model--- this only predefines the transformer vocabulary.)

The training directory is expected to contain subdirectories named by language family; within each one, the program will detect training files in Unimorph format (tab-separated lemma form feats).
  
To carry out multilingual training, use:
  
python -u script/byexample.py --run <NAME> --data ../2022InflectionST/part1-byfam --load_other <NAME_FROM_PREVIOUS_STEP>/model0/checkpoints --n_exemplars=dynamic --limit_train=20000 --s2s_max_num_epochs=100 --s2s_patience=100 --append_train --extra_features=all

To run a language:

#train model
python -u script/byexample.py --run sm_challenge_22/fine/ang --data ../2022InflectionST/part1-byfam/germanic/ang.train --devset ../2022InflectionST/part1-byfam/germanic/ang.dev --s2s_max_num_epochs=50 --s2s_patience=100 --load_other <PARENT_MODEL>/model0/checkpoints/ --allow_self_exemplar --extra_features=all

#write task output if you *didn't* use --extra_features to activate rule features
#python -u script/writeOutput.py ../2022InflectionST/part1-byfam/germanic/ang.dev sm_challenge_22/fine/ang/model0/predictions_dev.txt

#evaluation if you used --extra_features
  
echo "--------------- predicting ------------------"
  
python -u script/load_eval.py --run sm_challenge_22/fine/ang --devset sm_challenge_22/fine/ang/dev.txt
  
echo "--------------- creating featurized file for second pass -------------"
  
python -u script/usePredictedFeatures.py sm_challenge_22/fine/ang/ang-dev.txt/predictions_dev.txt
  
echo "--------------- predicting ------------------"
  
python -u script/load_eval.py --run sm_challenge_22/fine/ang --devset sm_challenge_22/fine/ang/ang-dev.txt/predictions_dev-predicted.txt 
  
echo "---------- write output --------------"
  
python -u script/writeOutput.py ../2022InflectionST/part1-byfam/germanic/ang.dev sm_challenge_22/fine/ang/ang-dev.txt-predictions_dev-predicted.txt/predictions_dev.txt > sm_challenge_22/fine/ang/writeOutput.log
  
You can create a runnable job script per language using create2020Jobs.py; this may require modification for your system.


-----------------------------------

How to derive predictions cued by a set of exemplars listed by
alignment microclass:

(0) Train a model as above

(1) Extract a set of target words microclass by microclass
    edit the script/finnishTargets.py script to indicate target cell(s)
	set the count cutoff (line 52)
	
    run python script/finnishTargets.py ../relevant/unimorph/Uralic/North_Sami/sme.trn morphomic_sme/probe/sme.dev

	(1st argument: Unimorph-format either train or dev (dev is less
	likely to give deceptive results due to overfitting, train has more data))
	(2nd argument: file to create (put in new directory somewhere))

(2) Translate the targets file into a set of instances with exemplars

	run python --run  morphomic_sme/probe/probe --data ../relevant/unimorph/Uralic/North_Sami/sme.trn --dev morphomic_sme/probe/sme.dev --nospace --extra_features=all
	(--run: a new directory to put working files in)
	(--data: the data to select exemplars from)
	(--dev: the target file you created in the previous step)
	(--nospace: collapse spaces in inflected forms)
	(--extra_features: use same as in model training)

(3) Decode the instances with exemplars

	$RUN = morphomic_sme
    python -u script/load_eval.py --run $RUN/probe/probe --devset $RUN/probe/probe/judgements/dev.txt --load_other $RUN/model0/checkpoints

(4) Decode the sequence probs as well (we won't need these, actually)

    python -u script/load_eval.py --run $RUN/probe/probe --devset $RUN/probe/probe/judgements/class.txt --sequence_probs --load_other $RUN/model0/checkpoints

(5) Generate the class-by-class report

	python script/finnishScore.py morphomic_sme/probe/probe/judgements-dev.txt/predictions_dev.txt morphomic_sme/probe/probe/judgements/key.txt morphomic_sme/probe/probe/not_actually_used > morphomic_sme/probe/probe/report

	(1st argument: decoded dev file)
	(2nd argument: key file indicating microclass per example)
	(3rd argument: technically an output file with some auxiliary
	data, actually commented out for reasons I don't remember)
	(output to stdout gets captured in the report)