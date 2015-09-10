local function splitColumns(str)
  local cols = utils.split(str, ' ')

  -- TODO check correctness
  if cols[3] == 'O' then cols[2] = cols[3] end -- other element
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
      word:gsub('.', function(c)
        currLen = currLen + 1
        x[currLen] = vocab[c]
        y[currLen] = vocab[tag..'-I']
      end)
      -- correct first and last char
      y[currLen + 1 - #word] = vocab[tag..'-B']
      y[currLen] = vocab[tag..'-E']
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
      if not utags[tag] then
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

local PosTester, testerParent = torch.class('PosTester', 'PosTranslator')

function PosTester:__init(rootDir)
  testerParent.__init(self, rootDir)

  local testFp = path.join(self.rootDir, 'test.txt')
  local totLen  = 0
  local rest, str, f

  -- calculate test file length, maybe I can simply use file size
  io.write('Calculating test text size...')
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
  print('ok')

  local x, y = makeCorpus(testFp, self.vocab, totLen)
  self.x = x
  self.y = y
end
