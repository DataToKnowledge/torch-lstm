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
require 'io.SeriesProcessor'
require 'io.OneHot'
require 'io.TimeHot'

-- models
require 'models.LSTM'
require 'models.LSTMN'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict from a model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model', 'model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-seedFile',"",'the file containing the observed timeseries values')
cmd:option('-length',5,'number of observations to sample')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

if not lfs.attributes(opt.model, 'mode') then
  print('Error the file ' .. opt.model .. ' does not exists, \n specify a right model file')
end
checkpoint = torch.load(opt.model)
merge(opt, checkpoint.opt)

-- has suitable gpu
if opt.gpuId >= 0 then
  io.write("Checking GPU...")
  if not opt.openCL then -- with CUDA
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if ok and ok2 then
      print('using CUDA on GPU' .. opt.gpuId)
      print('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuId -1 for sampling as well')
      cutorch.setDevice(opt.gpuId + 1)
      cutorch.manualSeed(opt.seed)
    else
      print('no suitable GPU but model was trained whit CUDA, stopping!')
      os.exit()
    end
  else
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if ok and ok2 then
      print('using OpenCL on GPU ' .. opt.gpuId)
      print('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuId -1 for sampling as well')
      cltorch.setDevice(opt.gpuId + 1)
      torch.manualSeed(opt.seed)
    else
      print('no suitable GPU but model was trained whit OpenCL, stopping!')
      os.exit()
    end
  end
end -- end has suitable gpu

protos = checkpoint.protos
protos.rnn:evaluate()


-- init the rnn state to all zeros
print('Creating an LSTM')
local currentState = {}

for layer = 1, opt.layersNumber do
  local initialH = torch.zeros(1, opt.layerSize):double()
  if opt.gpuId >= 0 then
    if opt.openCL then initialH = initialH:cl()
    else initialH = initialH:cuda() end
  end

  table.insert(currentState, initialH:clone())
  table.insert(currentState, initialH:clone())
end
local stateSize = #currentState

local seedNumbers = {}
for line in io.lines(opt.seedFile) do
  table.insert(seedNumbers, tonumber(line))
end

if #seedNumbers > 0 then
  print('seeding with ' .. #seedNumbers .. ' numbers')
  print('--------------------------')

  for i, n in pairs(seedNumbers) do
    local prevNum = torch.Tensor{n}
    print(n)
    if opt.gpuId >= 0 and opt.openCL == 0 then prevNum = prevNum:cuda() end
    if opt.gpuId >= 0 and opt.openCL == 1 then prevNum = prevNum:cl() end
    local lst = protos.rnn:forward{prevNum, unpack(currentState) }
    -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
    currentState = {}
    for i=1,stateSize do table.insert(currentState, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probabilities
  end
end

for i = 1, opt.length do
  local prevNum = prediction

  local lst =  protos.rnn:forward{prevNum, unpack(currentState)}
  currentState = {}
  for i=1,stateSize do table.insert(currentState, lst[i]) end
  prediction = lst[#lst] -- last element holds the log probabilities
  print(prevNum:max())
end
