local badTag = {
  ["("] = true,
  [")"] = true,
  [":"] = true,
  ["''"] = true,
  ["#"] = true,
  ["."] = true,
  ["``"] = true,
  ["$"] = true,
  [","] = true
}

local function splitColumns(str)
  local cols = utils.split(str, ' ')

  -- return word, tag and chunk
  return cols[1]:lower(), cols[2]:upper(), cols[3]:upper()
end

local function makeCorpus(inputFp, vocab, totLen)
  local buffer  = 2^14 -- 16k
  local rest, str, f

  local x, y = torch.ByteTensor(totLen), torch.ByteTensor(totLen)
  local currLen = 0
  f = io.open(inputFp,"r'")
  str, rest = f:read(buffer, '*line')
  while str do
    if rest then str = str .. rest .. '\n' end
    for _, l in pairs(utils.splitByLine(str)) do
      local word, tag = splitColumns(l)
      if #word == 1 then
        x[currLen] = vocab[c]
        y[currLen] = vocab[tag]
      else
        word:gsub('.', function(c)
          currLen = currLen + 1
          x[currLen] = vocab[c]
          y[currLen] = vocab[tag..'-I']
        end)
        -- correct first and last char
        y[currLen + 1 - #word] = vocab[tag..'-B']
        y[currLen] = vocab[tag..'-E']
      end -- end of char/tag storing
    end -- end for lines
    str, rest = f:read(buffer, '*line')
  end -- end while
  f:close()

  return x, y
end

local function textToTensor(inputFp)

  local timer = torch.Timer()

  print('Loading text file...')

  local uchars, ochars, utags, otags = {}, {}, {}, {}

  local buffer  = 2^14 -- 16k
  local totLen  = 0
  local rest, str, f

  io.write('Creating the vocabulary mapping...')

  f = io.open(inputFp, "r")
  str, rest = f:read(buffer, '*line')
  while str do
    if rest then str = str .. rest .. '\n' end
    for _, l in pairs(utils.splitByLine(str)) do
      local word, tag = splitColumns(l)
      word:gsub('.', function(c)
        if not uchars[c] then uchars[c] = true end
      end)
      if #word == 1 then
        utags[tag] = true
      else
        utags[tag..'-B'] = true
        utags[tag..'-I'] = true
        utags[tag..'-E'] = true
      end
      totLen = totLen + #word
    end -- end for lines
    str, rest = f:read(buffer, '*line')
  end -- end while
  f:close()

  -- sort chars
  for char in pairs(uchars) do
    ochars[#ochars + 1] = char
  end
  table.sort(ochars)

  -- sort tags
  for tag in pairs(utags) do
    otags[#otags + 1] = tag
  end
  table.sort(otags)

  --invert and merge the tables to create a map char/tag -> int
  local vocab = {}
  for i, c in ipairs(ochars) do
    vocab[c] = i
  end

  for i, t in ipairs(otags) do
    vocab[t] = i + #ochars
  end

  print("ok")

  -- construct the tensor of all the x
  io.write('Putting data into tensor...')

  local x, y = makeCorpus(inputFp, vocab, totLen)

  print("ok")

  return x, y, vocab
end

----------------------------------------------
--                  Loader                  --
----------------------------------------------

local PosLoader, loaderParent = torch.class('PosLoader' ,'TextLoader')

function PosLoader:loadData()
  local x, y, vocab

  local inputFp = path.join(self.rootDir, 'input.txt')
  local vocabFp = path.join(self.rootDir, 'vocab.t7')
  local dataFp = path.join(self.rootDir, 'data.t7')

  -- load x from disk if present or run preprocessing
  if self._needPreProcessing(inputFp, vocabFp, dataFp) then
    print(' Need preprocessing!')
    x, y, vocab = textToTensor(inputFp)
    io.write('Saving data...')
    torch.save(dataFp, {x=x, y=y})
    torch.save(vocabFp, vocab)
    print("ok.")
  else
    io.write("Loading saved files...")
    local data = torch.load(dataFp)
    x, y = data.x, data.y
    vocab = torch.load(vocabFp)
    print("ok")
  end

  -- get batch creation date
  self.createdAt = math.min(
  lfs.attributes(vocabFp).modification,
  lfs.attributes(dataFp).modification
  )

  -- count the vocabolary that will be the rnn input size
  vocabSize = 0
  for _ in pairs(vocab) do
    vocabSize = vocabSize + 1
  end

  return x, y, vocab, vocabSize
end

----------------------------------------------
--                Translator                --
----------------------------------------------
local PosTranslator, translatorParent = torch.class('PosTranslator', 'TextTranslator')

----------------------------------------------
--                  Tester                  --
----------------------------------------------

local function extractTagsFromVocab(vocab)
  -- need to generate tag list and map
  -- tags are in vocab bottom
  local tags, codes, map, i = {}, {}, {}, 0
  for t, c in pairs(vocab) do
    if(t:len() > 1) then -- is a tag
      local tag = t:sub(1, -3)
      if not codes[tag] then
        i = i + 1
        codes[tag] = i
        tags[i] = tag
      end
      map[c] = codes[tag]
    end
  end

  return tags, map
end

local PosTester, testerParent = torch.class('PosTester', 'PosTranslator')

function PosTester:__init(rootDir)
  testerParent.__init(self, rootDir)

  local testFp = path.join(self.rootDir, 'test.txt')
  local buffer  = 2^14 -- 16k
  local totLen  = 0
  local rest, str, f

  -- calculate test file length, maybe I can simply use file size
  -- io.write('Calculating test text size...')
  f = io.open(testFp, "r")
  str, rest = f:read(buffer, '*line')
  while str do
    if rest then str = str .. rest .. '\n' end
    for _, l in pairs(utils.splitByLine(str)) do
      local word = splitColumns(l)
      totLen = totLen + #word
    end -- end for lines
    str, rest = f:read(buffer, '*line')
  end -- end while
  f:close()
  -- print('ok')

  local x, y = makeCorpus(testFp, self.vocab, totLen)

  self.vocabSize = self.size
  self.size = totLen
  self.x = x
  self.y = y

  self:resetPredictions()

  local tags, map = extractTagsFromVocab(self.vocab)
  self.tagsMap = map

  self.charConfMatrix = optim.ConfusionMatrix(tags)
  self.wordConfMatrix = optim.ConfusionMatrix(tags)

  self.charConfMatrix:zero()
  self.wordConfMatrix:zero()
  -- confMatrix.mat is the confusion matrix
  -- rows contains targets (real label)
  -- cols contains predictions (computed label)

  self.cwb = 0 -- current word begin
  self.cwe = 0 -- current word end

  self.taggedWords = {}
  self.iwords = 0

end

function PosTester:resetPredictions()
  -- predicted output and index
  self.py = torch.ByteTensor(self.size)
  self.pyi = 0
end

function PosTester:addPrediction(y)
  self.pyi = self.pyi + 1
  self.py[self.pyi] = y

  local py = self.tagsMap[y] -- prediction
  local ty = self.tagsMap[self.y[self.pyi]] -- target

  -- char prediction
  self.charConfMatrix:add(py, ty)

  -- word prediction
  local tagName = self:reversedTranslate(self.y[self.pyi])
  local tagSuffix = tagName:sub(tagName:len()-1, -1) -- can be -B, -I or -E

  if tagSuffix == '-B' then
    self.cwb = self.pyi
  elseif tagSuffix == '-E' then
    self.cwe = self.pyi
    self:_addWordPrediction(ty)
  elseif tagSuffix ~= '-I' then -- is a ona char log word
    self.wordConfMatrix:add(py, ty)

    self.iwords = self.iwords + 1
    self.taggedWords[self.iwords] = self:reversedTranslate(self.x[self.pyi])..' '..self:reversedTranslate(py)
  end
end

function PosTester:_addWordPrediction(ty)
  local wordTagsRange = self.py:narrow(1, self.cwb, self.cwe - self.cwb + 1)
  local wordRange = self.x:narrow(1, self.cwb, self.cwe - self.cwb + 1)
  local word = ''

  wordRange:apply(function(x)
    word = word..self:reversedTranslate(x)
  end)

  local tc = {} -- tag count

  -- count tags
  wordTagsRange:apply(function(x)
    local y = self.tagsMap[x]
    if not tc[y] then tc[y] = 0 end
    tc[y] = tc[y] + 1
  end)

  local py = self.tagsMap[wordTagsRange[1]]
  for t, c in ipairs(tc) do
    if c >= tc[py] then py = t end
  end

  self.wordConfMatrix:add(py, ty)

  self.iwords = self.iwords + 1
  self.taggedWords[self.iwords] = word..' '..self:reversedTranslate(py)
end

function PosTester:precision() -- by column
  local classes = self.charConfMatrix.classes
  local charPrecisions, wordPrecisions = {},{}

  for i, t in pairs(classes) do
    charPrecisions[t] = self.charConfMatrix.mat[i][i] / self.charConfMatrix.mat:select(2,i):sum()
    wordPrecisions[t] = self.wordConfMatrix.mat[i][i] /self.wordConfMatrix.mat:select(2,i):sum()
  end

  return charPrecisions, wordPrecisions
end

function PosTester:recall() -- by row
  local classes = self.charConfMatrix.classes
  local charRecalls, wordRecalls = {},{}

  for i, t in pairs(classes) do
    charRecalls[t] = self.charConfMatrix.mat[i][i] / self.charConfMatrix.mat:select(1,i):sum()
    wordRecalls[t] = self.wordConfMatrix.mat[i][i] / self.wordConfMatrix.mat:select(1,i):sum()
  end

  return charRecalls, wordRecalls
end

function PosTester:__tostring__()
  return self.charConfMatrix:__tostring__()
end
