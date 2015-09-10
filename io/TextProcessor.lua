local function textToTensor(inputFp)

  local timer = torch.Timer()

  print('Loading text file...')

  local unordered, ordered = {}, {}

  local buffer  = 2^14 -- 16k
  local totLen  = 0
  local str, f

  io.write('Creating the vocabulary mapping...')

  f = io.open(inputFp, "r")
  str = f:read(buffer)
  while str do
    str:gsub('.', function(c)
      if not unordered[c] then unordered[c] = true end
    end)
    totLen = totLen + #str
    str = f:read(buffer)
  end
  f:close()

  -- we need to order char list before making vocabulary
  -- in this way in this way with the same dataset we get always the same vocabulary
  for char in pairs(unordered) do
    ordered[#ordered + 1] = char
  end
  table.sort(ordered)

  --invert the table to create a map char -> int
  local vocab = {}
  for i, c in ipairs(ordered) do
      vocab[c] = i
  end

  print("ok")

  -- construct the tensor of all the data
  io.write('Putting data into tensor...')

  local data = torch.ByteTensor(totLen)
  local currLen = 0
  f = io.open(inputFp,"r'")
  str = f:read(buffer)
  while str do
    str:gsub(".", function(c)
      currLen = currLen + 1
      data[currLen] = vocab[c]
    end)
    str = f:read(buffer)
  end
  f:close()

  print("ok")

  return data, vocab
end

----------------------------------------------
--                  Loader                  --
----------------------------------------------
local TextLoader, loaderParent = torch.class('TextLoader' ,'AbstractLoader')

function TextLoader:__init(rootDir, batchSize, seqLength, datasetSplit)
  loaderParent.__init(self, rootDir, batchSize, seqLength, datasetSplit)

  self.inputModule = OneHot
  self.criterion = nn.ClassNLLCriterion()
end

function TextLoader:loadData()
  local x, y, vocab

  local inputFp = path.join(self.rootDir, 'input.txt')
  local vocabFp = path.join(self.rootDir, 'vocab.t7')
  local dataFp = path.join(self.rootDir, 'data.t7')

  -- load data from disk if present or run preprocessing
  if self._needPreProcessing(inputFp, vocabFp, dataFp) then
    print(' Need preprocessing!')
    x, vocab = textToTensor(inputFp)
    io.write('Saving data...')
    torch.save(dataFp, x)
    torch.save(vocabFp, vocab)
    print("ok.")
  else
    io.write("Loading saved files...")
    x = torch.load(dataFp)
    vocab = torch.load(vocabFp)
    print("ok")
  end

  -- get batch creation date
  self.createdAt = math.min(
    lfs.attributes(vocabFp).modification,
    lfs.attributes(dataFp).modification
  )

  -- shift all data to generate y (eg. x = {1, 2, 3, 4, 5} => y = {2, 3, 4, 5, 1})
  y = x:clone()
  y:sub(1,-2):copy(x:sub(2,-1))
  y[-1] = x[1]

  -- count the vocabolary that will be the rnn input size
  local vocabSize = 0
  for _ in pairs(vocab) do
      vocabSize = vocabSize + 1
  end

  return x, y, vocab, vocabSize
end

function TextLoader._needPreProcessing(inputFp, vocabFp, dataFp)
  -- fetch file attributes to determine if we need to rerun preprocessing

  if not (path.exists(vocabFp) or path.exists(dataFp)) then
      -- prepro files do not exist, generate them
      io.write('Data files does not exists.')
      return true
  else
      -- check if the input file was modified since last time we
      -- ran the prepro. if so, we have to rerun the preprocessing
      local inputMd = lfs.attributes(inputFp).modification
      local vocabMd = lfs.attributes(vocabFp).modification
      local tensorMd = lfs.attributes(dataFp).modification
      if inputMd > vocabMd or inputMd > tensorMd then
          io.write('Data files detected as stale.')
          return true
      end
  end

  return false
end

----------------------------------------------
--                Translator                --
----------------------------------------------
local TextTranslator = torch.class('TextTranslator')

function TextTranslator:__init(rootDir)
  self.rootDir = rootDir

  local vocabFp = path.join(self.rootDir, 'vocab.t7')
  local vocab = torch.load(vocabFp)

  self.size = 0
  self.vocab = vocab
  self.invertedVocab = {}
  for c, i in pairs(self.vocab) do
    self.invertedVocab[i] = c
    self.size = self.size + 1
  end
end

function TextTranslator:translate(c)
  return self.vocab[c]
end

function TextTranslator:reversedTranslate(i)
  return self.invertedVocab[i]
end
