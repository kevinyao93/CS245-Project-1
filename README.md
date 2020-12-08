# CS245-Project-1
Seed-based Weakly Supervised Named Entity Recognition




Setup for Tensorflow-Named Entity Recognition
*Notes 
Utilizes tensorflow 1.13.1, but requires python 3.7.9 or lower. Can utilize pyenv to update local python enviornment.

Can either follow steps within the readme, but here's a summary
- Install tf_metrics
- Within the datafile (in our case testdata1) run make download-glove, will get the vector file
- An existing dataset exists within testdata1, you can change if wished, but if you just want to test the existing dataset, just directly run make build.

- Current implementation only uses lstf_crf_ema model, but for more models check out https://github.com/guillaumegenthial/tf_ner
- Go to the models/lstf_crf_ema folder, and run python main.py (This will train the bi-lstm + crf on the dataset)
- This will generate a results folder that will include the score inside and have the predictions for each train/test dataset.
- To get recall,precision and f1 score, run ../conlleval < results/score/{name}.preds.txt > results/score/score.{name}.metrics.txt

- To generate a predicted tags file from an input, go to interact.py and change the INPUT_FILE. Updated to direclty read in a text file and return an output_tags.txt file.
