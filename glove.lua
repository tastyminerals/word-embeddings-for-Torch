#!/usr/bin/env th
--[[ Convert glove embeddings .txt file to torch .t7
We can reduce the .t7 size extracting only those embeddings that exist in our training data vocab.
If text corpus path is provided extract its vocab and reduce word embeddings dict to corpus vocab.

Compatible GloVe embeddings:
  http://nlp.stanford.edu/data/glove.6B.zip
  http://nlp.stanford.edu/data/glove.840B.300d.zip

Usage:
  glove.lua filename.txt --> converts filename.txt to filename.t7
  glove.lua filename.txt [-r|--reduce] /path/to/corpus --> convert filename.txt to filename_adapted.t7
    with respect to corpus vocabulary
  glove.lua filename.txt [-t|--tokens] --> extract and print only tokens
]]

local path = require "pl.path"
local file = require "pl.file"
local dir = require "pl.dir"
local stringx = require "pl.stringx"
local utf8 = require "lua-utf8"

-- handle command line args
local txt,param,corpath = arg[1],arg[2],arg[3]
assert(path.isfile(txt),string.format('ERROR: "%s" file does not exist!',txt))
local outfile = path.splitext(txt)..'.t7'
if param == "-r" or param == "--reduce" then
  assert(path.isdir(corpath),string.format('ERROR: "%s" path does not exist!',corpath))
  outfile = path.splitext(txt)..'_adapted.t7'
end


--[[ Count tokens and return {token=cnt} map.
Ideally, your corpus should be already preprocessed (e.g. punctuation separated from words).]]
function build_vocab(text)
  local vocab = {}
  -- count tokens
  for word in utf8.gmatch(text,"%S+") do
    if vocab[word] then
      vocab[word] = vocab[word] + 1
    else
      vocab[word] = 1
    end
  end
  return vocab
end

-- read only tokens
function extract_glove_tokens()
  for line in io.lines(txt) do
    print(utf8.match(line,"%S+"))
  end
end

function reduce2corpus(corpath)
  -- read all .txt files in path
  local paths = dir.getfiles(corpath,"*.txt")
  local corpus = {}
  for i,p in pairs(paths) do
    table.insert(corpus,file.read(p))
  end
  -- merge train.txt, valid.txt data
  table.concat(corpus)
  local vocab = build_vocab(unpack(corpus))
  local size = 0
  for _ in pairs(vocab) do
    size = size + 1
  end
  return vocab, size
end

-- compute glove embeddings cnt and dim
function get_glove_stats(countwords)
  if countwords then
    print('counting words...')
    -- get embeddings cnt
    local cnt = 0
    for first in io.lines(txt) do
      cnt = cnt + 1
    end
    print('done')
  end
  -- get embeddings dim
  local dim = 300 -- default
  for first in io.lines(txt) do
    dim = #stringx.split(first)-1 -- -1 for token column
    break
  end
  return cnt,dim
end

-- convert glove .bin to .t7, reduce to vocabsize if given
function glove_convert(vocab,vocabsize)
  local words,dim = get_glove_stats(vocabsize)
  print('converting...')
  local w2i = {} -- word to index map
  local i2w = {} -- index to word map
  local tensor
  --local i2vec = {} -- index to vector map
  if vocabsize then
    tensor = torch.FloatTensor(vocabsize,dim)
  else
    tensor = torch.FloatTensor(words,dim)
  end
  -- use separate counters
  local row = 1
  local idx = 1
  for line in io.lines(txt) do
    local embed = stringx.split(line,' ')
    local word = table.remove(embed,1)
    local vecrep = torch.FloatTensor(embed)
    if vocabsize and vocab[word] and not w2i[word] then
      w2i[word] = row
      i2w[row] = word
      --i2vec[row] = vecrep
      tensor[{{row},{}}] = vecrep
      row = row + 1
    end
    if not vocabsize then
      w2i[word] = idx
      i2w[idx] = word
      tensor[{{idx},{}}] = vecrep
      idx = idx + 1
    end
  end
  print('done')
  -- write .t7
  local glove = {}
  glove.tensor = tensor
  glove.w2i = w2i
  glove.i2w = i2w
  --glove.i2vec = i2vec
  print(string.format('writing %s ...',outfile))
  torch.save(outfile,glove)
end

if param == '--tokens' or param == '-t' then
  extract_glove_tokens()
elseif corpath then
  glove_convert(reduce2corpus(corpath))
else
  glove_convert()
end
