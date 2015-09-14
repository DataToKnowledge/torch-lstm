-- require('mobdebug').start()
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
cmd:text('Predict from a model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model', 'model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-temperature',1,'temperature of sampling')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

--------------------------------------
--        Start loading data        --
--------------------------------------

if not lfs.attributes(opt.model, 'mode') then
  print('Error the file ' .. opt.model .. ' does not exists, \n specify a right model file')
end

-- since I cannot load checkpoint before loading cuda or cl libraries
-- I need to get the name of model file, then realize if the model was trained
-- with CPU, CUDA or openCL with a name convention
-- (the name contains either 'cpu', 'cuda' or 'opencl' string)

-- has suitable gpu
local fn = lfs.getFilename(opt.model)
if not string.match(fn, ".-(cpu).-$") then
  opt.gpuId = 0
  io.write("Checking GPU...")
  if string.match(fn, ".-(cuda).-$") then -- with CUDA
    opt.openCL = false
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
    opt.openCL = true
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
else
  opt.gpuId = -1
end -- end has suitable gpu

local archSetter = utils.getArchitectureSetter(opt)

io.write('Loading checkpoint...')
local checkpoint = torch.load(opt.model)
utils.merge(opt, checkpoint.opt)
print('ok')

local protos = checkpoint.protos
protos.rnn:evaluate()

-- init the tester
io.write('Loading test data...')
local tester = _G[opt.loader..'Tester'](opt.dataDir)
print('ok')

-- init the rnn state to all zeros
print('Creating an LSTM')
local currentState = {}

for layer = 1, opt.layersNumber do
  local initialH = torch.zeros(1, opt.layerSize):double()
  initialH = archSetter(initialH)

  table.insert(currentState, initialH:clone())
  table.insert(currentState, initialH:clone())
end
local stateSize = #currentState

--------------------------------------
--          Start  testing          --
--------------------------------------

print('Test started!')
tester.x:apply(function(x)
  input = archSetter( torch.Tensor{ x } )
  local lst = protos.rnn:forward{input, unpack(currentState) }

  -- update current state
  -- lst is a list of [state1,state2,..stateN,output], we want everything but last piece
  currentState = {}
  for i = 1, stateSize do table.insert(currentState, lst[i]) end
  prediction = lst[#lst]

  --log probabilities from the previous timestep
  if opt.sample == 0 then
    -- user argmax
    local _, _outcome = prediction:max(2)
    outcome = _outcome:resize(1)
  else
    -- se sampling
    prediction:div(opt.temperature)
    local probs = torch.exp(prediction):squeeze()
    probs:div(torch.sum(probs)) -- renormalize so probs sum to one
    outcome = torch.multinomial(probs:float(), 1):resize(1):float()
  end

  tester:addPrediction(outcome[1])
end)

torch.save('pos_cm.th7',tester.confMatrix)

print(tester.confMatrix)
