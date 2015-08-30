local AbstractLoader = torch.class('AbstractLoader')

--rootDir = the rootDirectory that contains a file input.txt
--batchSize = the number of stream of data to produce that will be trained in parallel
--seqLength = specifies the length of each chunk.
-- -- if `seqLength` is 20, then the gradient signal will never backpropagate more than 20 time steps, and the model might
-- -- not *find* dependencies longer than this length in number of characters.
-- datasetSplit divide the dataset into train, dev and test set {0.8, 0.1, 0.1}
function AbstractLoader:__init(rootDir, batchSize, seqLength, datasetSplit)
  self.rootDir = rootDir

  -- x is the indipendent variable (input)
  -- y is the dependent variable (output)
  -- vocab is a dictionary with byte => char mapping, can be nil
  local x, y, vocab, rnnInputSize = self:loadData()
  self.rnnInputSize = rnnInputSize

  -- cut off the end so that it divides evenly
  local len = x:size(1)
  if len % (batchSize * seqLength) ~= 0 then
    print('Cutting off end of data so that the batches/sequences divide evenly')
    len = batchSize * seqLength * math.floor(len / (batchSize * seqLength))
    x = x:sub(1, len)
    y = y:sub(1, len)
  end

  self.xBatches = x:view(batchSize, -1):split(seqLength, 2)  -- #rows = #batches
  self.yBatches = y:view(batchSize, -1):split(seqLength, 2)  -- #rows = #batches
  assert(#self.xBatches == #self.yBatches)

  self.nBatches = #self.xBatches

  -- lets try to be helpful here
  if self.nBatches < 50 then
    print('WARNING: less than 50 batches in the data in total? Looks like very small dataset.'..
      'You probably want to use smaller batchSize and/or seqLength.')
  end

  -- perform safety checks on datasetSplit
  assert(datasetSplit.train >= 0 and datasetSplit.train <= 1,
    'Bad split fraction ' .. datasetSplit.train .. ' for train, not between 0 and 1')
  assert(datasetSplit.val >= 0 and datasetSplit.val <= 1,
    'Bad split fraction ' .. datasetSplit.val .. ' for val, not between 0 and 1')
  assert(datasetSplit.test >= 0 and datasetSplit.test <= 1,
    'Bad split fraction ' .. datasetSplit.test .. ' for test, not between 0 and 1')

  self.nTrain = math.floor(self.nBatches * datasetSplit.train)
  if datasetSplit.test == 0 then
    -- catch a common special case where the user might not want a test set
    self.nVal = self.nBatches - self.nTrain
    self.nTest = 0
  else
    -- divide data to train/val and allocate rest to test
    self.nVal = math.floor(self.nBatches * datasetSplit.val)
    self.nTest = self.nBatches - self.nVal - self.nTrain -- the rest goes to test (to ensure this adds up exactly)
  end

  self.splitSizes = {train = self.nTrain, val = self.nVal, test = self.nTest}
  self.batchIx = {train = 0, val = 0, test = 0}

  print(string.format('Data load done. Number of data batches in train: %d, val: %d, test: %d', self.nTrain, self.nVal, self.nTest))
  collectgarbage()
end

function AbstractLoader:resetBatchPointer(splitName, batchIndex)
  batchIndex = batchIndex or 0
  self.batchIx[splitName] = batchIndex
end

function AbstractLoader:nextBatch(i)
  if self.splitSizes[i] == 0 then
    -- perform a check here to make sure the user isn't screwing something up
    print('ERROR. Code requested a batch for split ' .. i .. ', but this split has no data.')
    os.exit() -- crash violently
  end
  -- i is integer: 1 = train, 2 = val, 3 = test
  self.batchIx[i] = self.batchIx[i] + 1
  if self.batchIx[i] > self.splitSizes[i] then
    self.batchIx[i] = 1 -- cycle around to beginning
  end
  -- pull out the correct next batch
  local ix = self.batchIx[i]
  if i == 'val' then ix = ix + self.nTrain end -- offset by train set size
  if i == 'test' then ix = ix + self.nTrain + self.nVal end -- offset by train + val
  return self.xBatches[ix], self.yBatches[ix]
end
