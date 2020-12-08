"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor

LINE = 'John lives in New York'
INPUT_FILE = '../../data/testdata1/train.words.txt'

if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    with open(INPUT_FILE, 'r') as input, open('output.txt', 'w') as output:
        for input_line in input:
            words = [w.encode() for w in input_line.split()]
            nwords = len(words)
            predictions = predict_fn({'words': [words], 'nwords': [nwords]})
            preds = predictions['tags_ema']
            output_string = ''
            for item in preds[0]:
                output_string+='{} '.format(item.decode())
            output.write('{}\n'.format(output_string.strip()))
