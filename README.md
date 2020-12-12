# CS245-Project-1

Seed-based Weakly Supervised Named Entity Recognition

## Setup for Tensorflow-Named Entity Recognition

**This section is necessary whether you want to do a train/test on model directly or do everything from start !**

**Note:** 

Prerequisites as follows. Can utilize `pyenv` to update local python environment.

1. `tensorflow==1.13.1`
2. `python<=3.7.9` 

Can either follow steps within the readme, but here's a summary
- Install [`tf_metrics`](https://github.com/guillaumegenthial/tf_metrics) following the steps in `tf_ner/README.md`
- Enter the datafile directory (in our case `cd tf_ner/data/testdata1`) , then run `make download-glove`, will get the vector file.
- An existing dataset is in testdata1, you can change if wished, but if you just want to test the existing dataset, just directly run `make build`.
  - If a specific dataset is desired, optional inputs are available running `testa` and `train` by default. First build the vocab with `python make_vocab.py --test {testa} --train {train}`
  - The current main function of the model utilizes a set format to train and test the files.
  - Training the model: select the tag file and rename it: train.tags.txt, and name the text file: train.words.txt
  - Testing files: same vein as training, but use testa.tags.txt / testa.words.txt and testb.tags.txt / testb.words.txt
- Current implementation only uses lstf_crf_ema model, but for more models check out https://github.com/guillaumegenthial/tf_ner
- Go to the `tf_ner/models/lstm_crf_ema` folder, and run `python main.py` (This will train the bi-lstm + crf on the dataset)
  - If vocab was built on specific test set and not default, use the same commands above `python main.py --test {testa} --train {train}` to run.
- This will generate a results folder that will include the score inside and have the predictions for each train/test dataset.
- To get recall,precision and f1 score, run `../conlleval < results/score/{name}.tags.preds.txt > results/score/score.{name}.metrics.txt`
- To generate a predicted tags file from an input, go to `interact.py` and change the `INPUT_FILE`. Updated to directly read in a text file and return an `output_tags.txt` file.



## If you want to run tagged training files directly and do the test

In `tf_ner/data/testdata1`, there are already files that fit the input format of our model. Some of them are annotated using human annotation and some of them are based on entity set expansion. Here are the detail explanations of each file:

### CoNLL 2003:

`train_BERT_expand_tag_CoNLL.txt` : the tagged training set on CoNLL using BERT

`train_Spacy_expand_tag_CoNLL.txt` : the tagged training set on CoNLL using spaCy

`annotated_train_sentences_CoNLL.txt` : The training sentences of CoNLL

`annotated_train_tags_CoNLL.txt` : the original training tag of CoNLL.

To train with the entity set expansion result, use `train_BERT_expand_tag_CoNLL.txt` OR `train_Spacy_expand_tag_CoNLL.txt` with `annotated_train_sentences_CoNLL.txt`. To train with original sentence, use `annotated_train_sentences_CoNLL.txt` with `annotated_train_tags_CoNLL.txt`. 

To test the training model on CoNLL data, use `annotated_test_sentences_CoNLL.txt` with `annotated_test_tags_CoNLL.txt` as test inputs. 

### A subsection of Broad Twitter Corpus(section g):

To train with the entity set expansion result, use `train_twitter_g_sentences.txt` with `train_twitter_g_tags.txt`. To train with original sentences, use `annotated_train_sentences_twitter_g.txt` with `annotated_train_tags.txt_twitter_g.txt`

To test the training model on test set of section g, use `annotated_test_tags.txt_twitter_g.txt` and `annotated_test_sentences_twitter_g.txt` as test inputs.

### Result leveraging the plain twitter data:

To train the model on expanded entity set of plain twitter data only, please uncompress `0322_sentences_ and_tags.zip` and get `train_twitter_sentences_0322.txt` and `train_twitter_tags_0322.txt`

THEN, run

``` 
cat train_sentences.txt train_twitter_sentences_0322.txt > train_sentences_new.txt && cat train_broad_tag_by_entity.txt train_twitter_tags_0322.txt > train_tags_new.txt
```

to generate new sentences file `train_sentences_new.txt` and tag file `train_tags_new.txt` for training.

To test the training model on test set, use `test_tags.txt` and `test_sentences.txt` as test inputs.



## If you want to do everything from the start

### Generated phrases from Twitter


We put a example zip file (`2020-03-22_clean-hydrated.zip`) in the repository. This is the compressed result from one of our text file (`2020-03-22_clean-hydrated.txt`) as a example. We used AutoPhrase to mine the phrases from 7 text files like this. Due to the GitHub size restrictions, we can't upload everything related to AutoPhrase to make it run. However, the process of running AutoPhrase as follow:

**NOTE**

(Under folder `AutoPhrase`)

1. put your text file into `data/EN` (in this case would be `2020-03-22_clean-hydrated.txt`)
2. run `./auto_phrase.sh`
3. find your result in `model/DBLP`, notice that the `Autophrase.txt` is the file we need (this file includes salient phrases).
4. Do a parsing to remove the scores in each line, and remove all phrases with 3 or more words.
   
    **NOTE: because AutoPhrase requires many other tools related so we can't upload a whole repository of AutoPhrase files in here**
    **If you want to run AutoPhrase with full experience, please pull its original repo https://github.com/shangjingbo1226/AutoPhrase/tree/master/src and replace auto_phrase.sh with the one shown in our repository.**

We renamed those generated `AutoPhrase.txt`, and put them in a folder `autophrase_result`. This is the result doing step 1 to 3. We also include a parsing file to do step 4 in the same folder (`autophrase_result/parsing.py`).

We put all the phrases generated by AutoPhrase mining the Twitter texts into the folder `phrase_list`. Each file inside the folder is named after the date of the original full twitter text file (i.e. 0322 stands for tweets recorded on 03/22/2020)

### CoNLL 2003 data preparation

We use CoNLL 2003 data as a small trial on using Spacy or BERT. The folder `conll_2003` contains train/test/validation split and the python code for parsing the training/testing data to the format accepted by Bi-LSTM-CRF. 

`parsing_annotated_conll.py`  retains the original tag of the input data and parse it to 2 files, one for sentence and one for tag. As these 2 files are required for Bi-LSTM model. 

`parsing_conll_entity.py`  does the tagging based on the expanded entity set. This should be only used for CoNLL training set to cover the original tags.

### SpaCy Expansion

After the comparison between Bert and SpaCy, we decided to choose the word embeddings provided by spaCy for the following work. Folder `expansion` contains the python codes for entity expansion using spaCy word embeddings.

1. `SPACY_dictionary_expansion.py`: 

   - Before running the script, you can modify the `raw_data_path` , `process_data_path` ,  `dict_core_path` , `dict_path` , `candidate_path` variables at the top of the script according to your own needs, or follow the default settings used by us.

   - This is the python script for the final entity set expansion based on both the human-annotated Broad Twitter Corpus and plain twitter data.  The core dictionary is generated by combining the phrases in Broad Twitter Corpus which are annotated as LOC and the basic phrases in GeoName dictionary (specified by `dict_path`).  Since the location phrases in 'Broad' is not clean, we need to process the raw data (specified by `raw_data_path`)  and output the processed file specified by `process_data_path`.  The output file containing the combined core dictionary is specified by  `dict_core_path`, and the final expanded dictionary is specified by `candidate_path`.

   - The phrase lists used to expand the dictionary here are fixed in the code, they are plain text files in `phrase_list`. The existing output file is `./data/expanded_candidate.txt`

2. `SPACY_expansion.py`: 

   - Before running the script, you can modify the `phrase_lists_path` , `dict_path` , `candidate_path` variables at the top of the script according to your own needs. 

   - This is a more general python script that generates candidate phrases and add them to a specified core dictionary (specified by `dict_path`) based on the phrase list generated by AutoPhrase (specified by `phrase_lists_path`).  The output file containing the expanded dictionary is specified by `candidate_path`.

   - Dictionary expanded based on CoNLL 2003 dataset (`./data/SPACY_cand_conll.txt`) or section g of Broad Twitter Corpus (`./data/SPACY_cand_g_twitter.txt`) can be generated through this script.

### Bert Expansion

We only used Bert embeddings on CoNLL 2003 dataset in the small trial.

`expansion/Bert_conll_expansion.py`  does similar job as `SPACY_expansion.py` besides using Bert embeddings rather than spaCy. Similarly, `phrase_lists_path` , `dict_path` , `candidate_path` variables at the top of the script can be modified. 

Dictionary expanded based on CoNLL 2003 dataset (`./data/BERT_cand_conll.txt`) can be generated through this script.

### Final Input Generation

With the expanded dictionary in hand, we just need to tag the sentences based on it and generate the training data. Folder `expansion` also contains the python codes for tagging with expanded dictionary.

1. `tag_broad_training_entity.py` : use expanded dictionary `./data/expanded_candidate.txt` to tag the training part of Broad Twitter Corpus, output tag file is`train_broad_tag_by_entity.txt`

2. `parsing_hydrated_text.py` : use expanded dictionary `./data/expanded_candidate.txt` to tag the 0322 plain twitter texts, output tag file is `train_twitter_tags_0322.txt` , output sentence file is `train_twitter_sentences_0322.txt`

3. `parsing_conll_entity.py` : use expanded dictionary `./data/SPACY_cand_conll.txt` to tag the training part of CoNLL 2003 dataset, output tag file is `train_conll_spacy_tags.txt` , output sentence file is `train_conll_spacy_sentences.txt`
4. `parsing_entity_g.py` : use expanded dictionary `./data/SPACY_cand_g_twitter.txt` to tag the training part of Broad Twitter Corpus, output tag file is `train_twitter_g_tags.txt` , output sentence file is `train_twitter_g_sentences.txt`

### Model Training

All the codes related to model training are under the `tf_ner` folder.

1. Rename the training and testing files prepared in the previous step to `{train_name}.tags.txt`, `{train_name}.words.txt`,  `{test_name}.tags.txt`, `{test_name}.words.txt` then put then into a folder under `tf_ner/data` .

2. Train the model with the following commands:

   ```
   cd tf_ner/models/lstm_crf_ema/
   python main.py --test {test_name} --train {train_name}
   ```

   If `train`/`test` is not specified, the program will look for default files with name `{train}`/`{testa}` 

3. This will generate a `results` folder that will include the `score` folder inside with the predictions for each train/test dataset.

### Evaluations and Results 

1. To get recall, precision and f1 score, run

   ```
   ../conlleval < results/score/{name}.tags.preds.txt > results/score/score.{name}.metrics.txt
   ```

   This generates all the metrics in the `results/score/score.{name}.metrics.txt`

2. To generate a predicted tags file from an input, go to interact.py and change the INPUT_FILE. Updated to directly read in a text file and return an `output_tags.txt` file.



## Results from our experiments

Here are some of our experiments results:

### Test on ConLL 2003 dataset

|     Model      | Precision | Recall |   F1   |
| :------------: | :-------: | :----: | :----: |
|      BERT      |   0.32    | 0.862  | 0.4753 |
| SpaCy Word2vec |  0.4728   | 0.8653 | 0.6115 |

**Results of K-means Clustering on spaCy embeddings:**

Related codes can be found in `expansion/Clustering.ipynb`

![img](https://lh5.googleusercontent.com/JpLcBCi4uMNDDPas_CF5LnmS2bJLe5wtQEez0uJUHBfjls9Srl3LAWPIB3dKbozXG4Xq1JTr8fmHtzMFQMiW1JthmEyF_MGqC-OYqCt5HFTB9ge7l1cZafFK_zUOg8AESB3mCkQ)



### Using Twitter data to evaluate the system

|             Model              | Human Effort | Precision | Recall |   F1   |
| :----------------------------: | :----------: | :-------: | :----: | :----: |
|            Bi-LSTM             |  annotated   |  0.7645   | 0.5027 | 0.6066 |
|      Dictionary matching       |     none     |  0.7747   | 0.3832 | 0.5127 |
| Bi-LSTM + entity set expansion |     none     |  0.7213   | 0.4783 | 0.5752 |

###  Leveraging the plain twitter data

|             Model              | Human Effort | Precision | Recall |   F1   |
| :----------------------------: | :----------: | :-------: | :----: | :----: |
|            Bi-LSTM             |  annotated   |  0.7232   | 0.788  | 0.7542 |
| Bi-LSTM + entity set expansion |     none     |  0.3636   | 0.4946 | 0.4191 |

## Miscellaneous
If you would like to see the human annotated locations in braod training set, check out `locations_from_train.txt`
