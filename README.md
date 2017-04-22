# word-embeddings-for-Torch
Pretrained word embeddings (aka word vectors) take a lot of space and memory to process. Moreover, many of these pretrained embeddings come in **.bin** format with different data layouts or structure (GloVe vs word2vec). These scripts allow you to convert **.bin** embeddings to **.t7** format for easy load and use in [Torch](http://torch.ch/). In addition you can reduce the size of **.t7** file by fitting to your training corpus vocabulary.

### Usage
- Convert all word2vec embeddings to **.t7**
```bash
th word2vectot7.lua GoogleNews-vectors-negative300.bin  
```

- Extract pretrained embeddings only for tokens in your training corpus
```bash
bin2t7.lua filename.bin -r /path/to/corpus
bin2t7.lua filename.bin --reduce /path/to/corpus
```
If your `/path/to/corpus` contains several **.txt** files (`train.txt`, `valid.txt`, `test.txt`) then the script will read each and create a cumulative vocabulary.

Currently only [word2vec](https://code.google.com/archive/p/word2vec/) converter is ready.

