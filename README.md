# word-embeddings-for-Torch
Pretrained word embeddings (aka word vectors) take a lot of space and memory to process. Moreover, many of these pretrained embeddings come in **.bin** format with different data layouts or structure (GloVe vs word2vec). These scripts allow you to convert **.bin** embeddings to **.t7** format for easy load and use in [Torch](http://torch.ch/). In addition you can reduce the size of **.t7** file by fitting to your training corpus vocabulary.

### Usage
- Convert all word2vec embeddings to **.t7**
```bash
th word2vec.lua GoogleNews-vectors-negative300.bin  
```

- Extract and convert to **.t7** pretrained embeddings only for tokens in your training corpus
```bash
th word2vec.lua filename.bin -r /path/to/corpus
```

- Extract and print only tokens
```bash
th word2vec.lua filename.bin -t
```
If your `/path/to/corpus` contains several **.txt** files (`train.txt`, `valid.txt`, `test.txt`) then the script will read each and create a cumulative vocabulary.

### Word Embeddings & Available Converters
- Download [Google News (word2vec)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), converter [word2vec](https://raw.githubusercontent.com/tastyminerals/word-embeddings-for-Torch/master/word2vec.lua)

- Download [GloVe - Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download)](http://nlp.stanford.edu/data/glove.6B.zip)

- Download [GloVe - Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)](http://nlp.stanford.edu/data/glove.840B.300d.zip()
