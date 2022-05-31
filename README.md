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
