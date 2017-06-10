# word-embeddings-for-Torch
Pretrained word embeddings (aka word vectors) take a lot of space and memory to process. Moreover, many of these pretrained embeddings come in **.bin** format with different data layouts or structure (GloVe vs word2vec). These scripts allow you to convert **.bin** embeddings to **.t7** format for easy load and use in [Torch](http://torch.ch/). In addition you can reduce the size of **.t7** file by fitting to your training corpus vocabulary.

The script requires ~4.5GB free RAM unless you use `[-r|--reduce]` parameter.

### Torch (.t7) file output format
```lua
{
  i2w -- {idx: token}
  tensor -- FloatTensor - size: vocabsize x 300
  w2i -- {token: idx}
}
```

### Usage
Convert all word2vec embeddings to **.t7**
```bash
th word2vec.lua GoogleNews-vectors-negative300.bin  
```

Extract and convert to **.t7** only for tokens in your training corpus
```bash
th word2vec.lua filename.bin -r /path/to/corpus
```

Extract and print tokens only
```bash
th word2vec.lua filename.bin -t
```

Extract and print tokens + their corresponding vector represenataions to stdout
```bash
th word2vec.lua filename.bin -tv
```

If your `/path/to/corpus` contains several **.txt** files (`train.txt`, `valid.txt`, `test.txt`) then the script will read each and create a cumulative vocabulary.

### Available Converters
- [word2vec converter](https://raw.githubusercontent.com/tastyminerals/word-embeddings-for-Torch/master/word2vec.lua)
- [GloVe converter](https://raw.githubusercontent.com/tastyminerals/word-embeddings-for-Torch/master/glove.lua)

### Word Embeddings
- Download [Google News (word2vec)](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) (**3.4 GB**)

- Download [GloVe - Wikipedia 2014 + Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, **822 MB**)

- Download [GloVe - Common Crawl](http://nlp.stanford.edu/data/glove.840B.300d.zip) (840B tokens, 2.2M vocab, cased, 300d vectors, **2.03 GB**)]

### Loading Torch (.t7) files in Python
Check out [python-torchfile](https://github.com/bshillingford/python-torchfile).

