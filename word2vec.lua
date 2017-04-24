#!/usr/bin/env th
--[[ Convert word2vec embeddings .bin file to torch .t7
We can reduce the .t7 size extracting only those embeddings that exist in our training data vocab.
If text corpus path is provided extract its vocab and reduce word embeddings dict to corpus vocab.
The script requires ~4.5GB free RAM unless you use --reduce parameter.

.bin to .t7 conversion code is taken from:
  https://github.com/rotmanmi/word2vec.torch/blob/master/bintot7.lua

Compatible word2vec embeddings:
  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

Usage:
  word2vec.lua filename.bin --> converts filename.bin to filename.t7
  word2vec.lua filename.bin [-r|--reduce] /path/to/corpus --> convert filename.bin to filename_adapted.t7
    with respect to corpus vocabulary
  word2vec.lua filename.bin [-t|--tokens] --> extract and print only tokens

.t7 file table structure:
  i2w -- {idx: token}
  tensor -- FloatTensor - size: vocabsize x 300
  w2i -- {token: idx}
]]

local path = require "pl.path"
local file = require "pl.file"
local dir = require "pl.dir"
local utf8 = require "lua-utf8"

-- handle command line args
local bin,param,corpath = arg[1],arg[2],arg[3]
assert(path.isfile(bin),string.format('ERROR: "%s" file does not exist!',bin))
local outfile = path.splitext(bin)..'.t7'
if param == "-r" or param == "--reduce" then
  assert(path.isdir(corpath),string.format('ERROR: "%s" path does not exist!',corpath))
  outfile = path.splitext(bin)..'_adapted.t7'
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

function read_word(diskfile,max_w)
  local str = {}
  for i = 1,(max_w or 50) do
    local char = diskfile:readChar()
    if char == 32 or char == 10 or char == 0 then
      break
    else
      str[#str+1] = char
    end
  end
  str = torch.CharStorage(str)
  return str:string()
end

function extract_word2vec_tokens()
  local diskfile = torch.DiskFile(bin,'r')
  local max_w = 50
  -- reading header
  diskfile:ascii()
  -- read .bin sizes
  local words = diskfile:readInt()
  local dim = diskfile:readInt() -- word2vec uses 300
  -- reading contents into tensor
  diskfile:binary()
  local row = 1 -- reduced .t7 needs a separate index
  for i = 1,words do
    local word = read_word(diskfile,max_w)
    diskfile:readFloat(300)
    print(word)
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

-- convert word2vec .bin to .t7, reduce to vocabsize if given
function word2vec_convert(vocab,vocabsize)
  local diskfile = torch.DiskFile(bin,'r')
  local max_w = 50
  -- reading header
  diskfile:ascii()
  -- read .bin sizes
  local words = diskfile:readInt()
  local dim = diskfile:readInt() -- word2vec uses 300
  local w2i = {} -- word to index map
  local i2w = {} -- index to word map
  local tensor
  --local i2vec = {} -- index to vector map
  if vocabsize then
    tensor = torch.FloatTensor(vocabsize,dim)
  else
    tensor = torch.FloatTensor(words,dim)
  end
  -- reading contents into tensor
  diskfile:binary()
  local row = 1 -- reduced .t7 needs a separate index
  for i = 1,words do
    local word = read_word(diskfile,max_w)
    local vecrep = diskfile:readFloat(300)
    vecrep = torch.FloatTensor(vecrep)
    local norm = torch.norm(vecrep,2)
    if norm ~= 0 then vecrep:div(norm) end
    -- reducing contents to vocab
    if vocabsize and vocab[word] and not w2i[word] then
      w2i[word] = row
      i2w[row] = string.lower(word) -- you'd better be working with lowercased training data
      --i2vec[row] = vecrep
      tensor[{{row},{}}] = vecrep
      row = row + 1
    end
    if not vocabsize then
      w2i[word] = i
      i2w[i] = string.lower(word) -- you'd better be working with lowercased training data
      --i2vec[i] = vecrep
      tensor[{{i},{}}] = vecrep
    end
  end
  -- write .t7
  local word2vec = {}
  word2vec.tensor = tensor
  word2vec.w2i = w2i
  word2vec.i2w = i2w
  --word2vec.i2vec = i2vec
  print(string.format('writing %s ...',outfile))
  torch.save(outfile,word2vec)
end

if param == '-t' or param == '--tokens' then
  extract_word2vec_tokens()
elseif corpath then
  word2vec_convert(reduce2corpus(corpath))
else
  word2vec_convert()
end
