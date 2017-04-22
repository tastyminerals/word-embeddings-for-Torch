#!/usr/bin/env th
--[[ Convert glove embeddings .bin file to torch .t7
We can reduce the .t7 size extracting only those embeddings that exist in our training data vocab.
If text corpus path is provided extract its vocab and reduce word embeddings dict to corpus vocab.
The script requires ~4.5GB free RAM unless you use --reduce parameter.

.bin to .t7 conversion code is taken from:

  https://github.com/rotmanmi/glove.torch

Usage:
  glove.lua filename.bin --> converts filename.bin to filename.t7
  glove.lua filename.bin [-r|--reduce] /path/to/corpus --> convert filename.bin to filename_adapted.t7
    with respect to corpus vocabulary

]]

local path = require "pl.path"
local file = require "pl.file"
local dir = require "pl.dir"
local utf8 = require "lua-utf8"

-- handle command line args
local bin,param,corpath = arg[1],arg[2],arg[3]
assert(path.isfile(bin),"ERROR: file does not exist!")
local outfile = path.splitext(bin)..'.t7'
if param == "-r" or param == "--reduce" then
  assert(path.isdir(corpath), "ERROR: path does not exist!")
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

function glove_dims(diskfile)
  local encodingsize = -1
  local ctr = 0
  for line in io.lines(diskfile) do
    if ctr == 0 then
      for i in string.gmatch(line, "%S+") do
        encodingsize = encodingsize + 1
      end
    end
    ctr = ctr + 1
  end
  return ctr,encodingsize
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

-- convert glove .bin to .t7, reduce to vocabsize if given
function glove_convert(vocab,vocabsize)
  local diskfile = torch.DiskFile(bin,'r')
  local words,dim = glove_dims(diskfile)
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
  local idx = 1
  local row = 1 -- reduced .t7 needs a separate index
  for line in io.lines(opt.binfilename) do
    local vecrep = {}
    for i in string.gmatch(line, "%S+") do
      table.insert(vecrep, i)
    end
    str = vecrep[1]
    table.remove(vecrep,1)
    vecrep = torch.FloatTensor(vecrep)

    local norm = torch.norm(vecrep,2)
    if norm ~= 0 then vecrep:div(norm) end

    if vocabsize and vocab[word] and not w2i[word] then
      w2i[word] = row
      i2w[row] = word
      --i2vec[row] = vecrep
      tensor[{{row},{}}] = vecrep
      row = row + 1
    end
    if not vocabsize then
      w2vvocab[str] = idx
      v2wvocab[idx] = str
      tensor[{{idx},{}}] = vecrep
      idx = idx + 1
    end
  end

  -- write .t7
  local glove = {}
  glove.tensor = tensor
  glove.w2i = w2i
  glove.i2w = i2w
  --glove.i2vec = i2vec
  print(string.format('writing %s ...',outfile))
  torch.save(outfile,glove)
end


if corpath then
  glove_convert(reduce2corpus(corpath))
else
  glove_convert()
end
