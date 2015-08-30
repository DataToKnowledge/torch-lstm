-- require installed libraries
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

-- utils
require 'utils.functions'
require 'utils.lfs'

-- preprocessing
require 'io.AbstractProcessor'
require 'io.TextProcessor'
require 'io.PosProcessor'
require 'io.OneHot'

-- models
require 'models.LSTM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an LSTM (long-short term memory) rnn')
cmd:text()
cmd:text('opt')
-- data
cmd:option('-loader', 'Pos', 'the data loader to be used during training (eg. Pos, Text, ecc...)')
cmd:option('-dataDir', 'data/postagging', 'data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-layerSize', 128, 'size of LSTM internal state')
cmd:option('-layersNumber', 2, 'number of layers in the LSTM')
-- optimization
cmd:option('-learningRate', 2e-3, 'learning rate')
cmd:option('-learningRateDecay', 0.97, 'learning rate decay')
cmd:option('-learningRateDecayAfter', 10, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decayRate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seqLength', 50, 'number of timesteps to unroll for')
cmd:option('-batchSize', 50, 'number of sequences to train on in parallel')
cmd:option('-maxEpochs', 50, 'number of full passes through the training data')
cmd:option('-gradClip', 5, 'clip gradients at this value')
cmd:option('-trainFrac', 0.95, 'fraction of data that goes into train set')
cmd:option('-valFrac', 0.5, 'fraction of data that goes into validation set')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-printEvery', 10, 'how many steps/minibatches between printing out the loss')
cmd:option('-evalValEvery', 500, 'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpointDir', 'cp', 'output directory where checkpoints get written, relative to dataDir')
cmd:option('-saveFileName', 'lstm', 'filename to autosave the checkpoint to. Will be inside checkpointDir/')
cmd:option('-initFrom', '', 'checkpoint file from which initialize network parameters, relative to checkpointDir')
cmd:option('-noResume', false, 'whether resume or not from last checkpoint')
-- GPU/CPU
cmd:option('-gpuId', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-openCL', false, 'use OpenCL (instead of CUDA)')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- has suitable gpu
if opt.gpuId >= 0 then
  io.write("Checking GPU...")
  if not opt.openCL then -- with CUDA
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if ok and ok2 then
      print('using CUDA on GPU' .. opt.gpuId)
      cutorch.setDevice(opt.gpuId + 1)
      cutorch.manualSeed(opt.seed)
    else
      opt.openCL = true -- try with OpenCL
    end
  end

  if opt.openCL then -- with OpenCL
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if ok and ok2 then
      print('using OpenCL on GPU ' .. opt.gpuId)
      cltorch.setDevice(opt.gpuId + 1)
      torch.manualSeed(opt.seed)
    else
      print('no suitable GPU, falling back on CPU mode')
      opt.gpuId = -1 -- overwrite user setting
    end
  end
end -- end has suitable gpu

-- train / validation / test split for data, in fractions
local sizes = {
  test = math.max(0, 1 - (opt.trainFrac + opt.valFrac)), -- calculate test for integrity
  train = opt.trainFrac,
  val = opt.valFrac
}

local loader = _G[opt.loader..'Loader'](opt.dataDir, opt.batchSize, opt.seqLength, sizes)

-- create directory for the check points
opt.checkpointDir = path.join(opt.dataDir, opt.checkpointDir)
if not path.exists(opt.checkpointDir) then
  lfs.mkdir(opt.checkpointDir)
end

protos = {}

-- candidate checkpoint file
if string.len(opt.initFrom) == 0 then
  local mrc = lfs.mostRecent(opt.checkpointDir)
  if mrc ~= nil then opt.initFrom = mrc
  else opt.noResume = true end
end
opt.initFrom = path.join(opt.checkpointDir, opt.initFrom)

if opt.noResume then -- check if it can restore model from checkpoints
  io.write('Creating an LSTM with ' .. opt.layersNumber .. ' layers...')
  protos.rnn = LSTM(loader.rnnInputSize, loader.inputModule, opt.layerSize, opt.layersNumber, opt.dropout)
  protos.criterion = loader.criterion
  print('ok')
else -- define the model for one timestep and then clone for the
  io.write('Trying to resume model from a checkpoint in ' .. opt.initFrom .. '...')
  -- check validity
  -- if loader data more recent then checkpoint, this is not valid anymore
  if loader.createdAt > lfs.attributes(from).modification then
    print('The checkpoints are not valid, please delete '.. opt.checkpointDir ..' and restart the training')
    os.exit()
  end

  local checkpoint = torch.load(from)
  opt.layerSize = checkpoint.opt.layerSize
  opt.layersNumber = checkpoint.opt.layersNumber
  protos = checkpoint.protos
  print('ok.')
end

-- initialize the state of the cell/hidden states
-- with CUDA or OpenCL if present
local initialState = {}
for layer = 1, opt.layersNumber do
  local initialH = torch.zeros(opt.batchSize, opt.layerSize)
  if opt.gpuId >= 0 then
    if opt.openCL then initialH = initialH:cl()
    else initialH = initialH:cuda() end
  end

  table.insert(initialState, initialH:clone())
  table.insert(initialState, initialH:clone())
end

if opt.gpuId >= 0 then
  if opt.openCL then
    for _, v in pairs(protos) do v:cl() end
  else
    for _, v in pairs(protos) do v:cuda() end
  end
end

-- put al the network parameters (weights) into a flattened parameter tensor
params, gradParams = combineParams(protos.rnn)

if not opt.noResume then
  params:uniform(-0.008, 0.08)
end

print('There are ' .. params:nElement() .. ' paramters in the model')
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do
  print('Cloning ' .. name)
  clones[name] = clone(proto, opt.seqLength, not proto.parameters)
end

local initialStateGlobal = cloneList(initialState)

-- prepare next batch from loader initializing CUDA or OpenCL
function prepareNextBatch(splitName)
  local x, y = loader:nextBatch(splitName)
  if opt.gpuId >= 0 then
    if opt.openCL then
      x = x:cl()
      y = y:cl()
    else
      x = x:float():cuda()
      y = y:float():cuda()
    end
  end

  return x, y
end

-- evalutate the loss over an entire split of size batchSize doing a forward pass
function evalSplit(splitName, maxBatches)
  print('Evaluating the loss over the ' .. splitName .. ' split')
  local n = loader.splitSizes[splitName]
  if maxBatches ~= nil then n = math.min(maxBatches, n) end

  loader:resetBatchPointer(splitName)
  local loss = 0
  local networkState = { [0] = initialState }

  -- iterate over the batches in the split
  for i = 1, n do
    --fecth a batch
    local x, y = prepareNextBatch(splitName)

    -- forward pass
    for t = 1, opt.seqLength do
      clones.rnn[t]:evaluate()
      local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(networkState[t-1]) }
      networkState[t] = {}
      for i = 1, #initialState do table.insert(networkState[t], lst[i]) end
      prediction = lst[#lst]
      loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
    end

    -- carry over the lstm state
    networkState[0] = networkState[#networkState]
  end

  loss = loss / opt.seqLength / n
  return loss
end

-- do a forward and a backward pass
function fbPass(x)
  if x ~= params then params:copy(x) end
  gradParams:zero()

  -- get minibatch and ship to gpu --
  local x, y = prepareNextBatch('train')

  -- forward pass
  local rnnState = {[0] = initialStateGlobal }
  local predictions = {}
  local loss = 0
  for t = 1, opt.seqLength do -- time unroll
    clones.rnn[t]:training()
    local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnnState[t-1]) }
    rnnState[t] = {}
    -- extract the state, without output
    for i = 1, #initialState do table.insert(rnnState[t], lst[i]) end
    -- get the last element with the prediction
    predictions[t] = lst[#lst]
    loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
  end
  loss = loss / opt.seqLength

  -- backward pass
  -- initialize gradient at time t to be zeros
  -- d stands for derivative
  local dRnnState = {[opt.seqLength] = cloneList(initialState, true)} -- true clone also the zero
  -- a for with decrement
  for t = opt.seqLength, 1, -1 do -- reversed loop
    -- backprop through loss, and softmax/linear
    local dOutputT = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
    table.insert(dRnnState[t], dOutputT)
    local dLst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnnState[t - 1])}, dRnnState[t])
    dRnnState[t-1] = {}
    for k,v in pairs(dLst) do
      -- set the gradient for all the hidden layers
      if k > 1 then dRnnState[t-1][k-1] = v end
    end
  end

  -- transfer final state to initial state
  initialStateGlobal = rnnState[#rnnState]
  -- clip the gradient element-wise
  gradParams:clamp(-opt.gradClip, opt.gradClip)
  return loss, gradParams
end

-- start the optionization from here
local optimState = {learningRate = opt.learningRate, alpha = opt.decayRate }
local iterations = opt.maxEpochs * loader.nTrain
local losses = {train = {}, val = {}}
local loss0 = nil

for i = 1, iterations do
  local epoch = i / loader.nTrain

  local timer = torch.Timer()
  -- the function to evaluate, the current parameter vector and a the table of parametes
  -- RmsProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
  local _, loss = optim.rmsprop(fbPass, params, optimState)
  local time = timer:time().real

  -- pop the loss in the list of train loss
  losses.train[i] = loss[1]

  -- exponential learning decay rate
  if i % loader.nTrain == 0 and opt.learningRateDecay < 1 then
    if epoch >= opt.learningRateDecayAfter then
      local decayFactor = opt.learningRateDecay
      optimState.learningRate = optimState.learningRate * decayFactor
      print("decayed learning rate by a factor from " .. optimState.learningRate .. ' to ' .. decayFactor)
    end
  end

  -- evaluate the loss on the validation set and save a checkpoint
  if i % opt.evalValEvery == 0 or i == iterations then
    losses.val[i] = evalSplit('val')

    -- save a checkpoint
    local fileName = string.format('%s/lm_%s_epoch%.2f_loss%.4f.t7',
    opt.checkpointDir, opt.saveFileName, epoch, losses.val[i])
    print('saving a checkoint to ' .. fileName)
    torch.save(fileName, {
      protos = protos,
      losses = losses,
      epoch = epoch,
      opt = opt,
      i = i
    })
  end

  if i % opt.printEvery == 0 then
    print(string.format("%d/%d (epoch %.3f), train loss = %6.8f, grad/param = norm = %6.4e, time/batch = %.2fs",
    i, iterations, epoch, losses.train[i], gradParams:norm() / params:norm(), time))
  end

  if i % 10 == 0 then collectgarbage() end

  -- handle early stopping if things are going really bad
  if loss[1] ~= loss[1] then -- loss[1] ~= loss[1] test if loss[1] is NaN
    print('Loss is NaN. '..
    'This usually indicates a bug. '..
    'Please check the issues page for existing issues, or create a new issue, if none exist. '..
    'Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
    break -- halt
  end

  if loss0 == nil then loss0 = loss[1] end
  if loss[1] > loss0 * 3 then
    print('loss is exploding, aborting')
    break
  end
end
