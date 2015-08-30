require 'io.textUtils'

-- copy all the data in the file to a tensor of Doubles
local function textToTensor(inputPath)
  local timer = torch.Timer()
  print('Loading input file ' .. inputPath)

  local bufferSize = 2^14 -- 16k
  local dataTable = {}

  io.write("Loading data from " .. inputPath .. ' ... ')
  local f = io.open(inputPath, "r")
  -- read from the file 16k characters per time + the rest till the end of line
  local buffer, rest = f:read(bufferSize, '*line')
  -- while the buffer is not nil
  while buffer do
    -- add the rest of the line
    if rest then buffer = buffer .. rest .. '\n' end
    for _, l in pairs(splitByLine(buffer)) do
      table.insert(dataTable, tonumber(l))
    end
    buffer, rest = f:read(bufferSize, '*line')
  end
  f:close()
  print("ok")

  io.write("Putting data to tensor ... ")
  local data = torch.DoubleTensor(dataTable)
  print("ok")
  return data
end


local SeriesLoader, parent = torch.class('SeriesLoader', 'AbstractLoader')

-- define the data loader with the input and output model for the network
function SeriesLoader:__init(rootDir, batchSize, seqLength, datasetSplit)
  parent.__init(self, rootDir, batchSize, seqLength, datasetSplit)

  self.inputModule = TimeHot
  self.criterion = nn.MSECriterion() -- or DistKLDivCriterion()
end

function SeriesLoader:loadData()
  local x, y

  local inputPath = path.join(self.rootDir, 'input.txt')
  local dataPath = path.join(self.rootDir, 'data.t7')

  -- load data from the disk
  if self._needPreProcessing(inputPath, dataPath) then
    x = textToTensor(inputPath)
    io.write("Saving the data to the file " .. dataPath .. " ... ")
    torch.save(dataPath, x)
    print("ok")
  else
    io.write('Loading data from save file ' .. dataPath .. " ... ")
    x = torch.load(dataPath)
    print("ok")
  end

  -- get data creation time
  self.createdAt = math.min(
    lfs.attributes(inputPath).modification,
    lfs.attributes(dataPath).modification
  )

  -- create y data
  y = x:clone()
  y:sub(1,-2):copy(x:sub(2,-1))
  y[-1] = x[1]

  local vocab = nil
  local rnnInputSize = 1
  return x, y, vocab, rnnInputSize
end

function SeriesLoader._needPreProcessing(inputPath, dataPath)
  -- fetch file attributes to determine if we need to rerun preprocessing

  if not path.exists(dataPath) then
    print('Data files does not exists.')
    return true
  else
    local inputMod = lfs.attributes(inputPath).modification
    local dataMod = lfs.attributes(dataPath).modification
    if inputMod > dataMod then
      print('detected data files are not correct')
      return true
    end
  end
    -- otherwise
    return false
end
