require 'nn'
require 'nngraph'

-- see https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf

-- nn.Identity()() creates a module that returns whatever is input to it as output without transformation
-- nn.Linear(inputSize,outputSize)(whatTranform) applies a linear transformation to the incoming data, i.e. y = Ax + b

local LSTM, parent = torch.class('LSTM','nn.gModule')

function LSTM:__init(inputSize, inputModule, layerSize, layersNumber, dropout)
  -- During training, dropout masks parts of the input using binary samples
  -- from a bernoulli distribution. Each input element has a probability of p
  -- of being dropped, i.e having its commensurate output element be zero.
  dropout = dropout or 0

  -- there will be 2*layersNumber+1 inputs
  local inputs, outputs = {}, {}

  table.insert(inputs, nn.Identity()()) -- x, the input layer
  for layer = 1,layersNumber do -- for every hidden layers
    -- I have the two inputs
    table.insert(inputs, nn.Identity()()) -- prevC[layer]
    table.insert(inputs, nn.Identity()()) -- prevH[layer]
  end

  local x, layerInputSize
  for layer = 1,layersNumber do
    -- c,h from previos timesteps
    local prevC = inputs[layer*2] -- the cells states
    local prevH = inputs[layer*2+1] -- the hidden nodes states

    -- the input to this layer
    if layer == 1 then
      x = inputModule(inputSize)(inputs[1])
      layerInputSize = inputSize
    else
      x = outputs[(layer-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      layerInputSize = layerSize
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(layerInputSize, 4 * layerSize)(x)
    local h2h = nn.Linear(layerSize, 4 * layerSize)(prevH)
    local allInputSums = nn.CAddTable()({i2h, h2h})

    -- decode the gates
    local sigmoidChunk = nn.Sigmoid()(nn.Narrow(2, 1, 3 * layerSize)(allInputSums))
    -- it returns the narrowed version of the above tensor.
    -- it narrows dimension 2 from index 1 to index 1+ (layerSize -1)
    local inputGate    = nn.Narrow(2, 1, layerSize)                 (sigmoidChunk)
    local forgetGate   = nn.Narrow(2, layerSize + 1, layerSize)     (sigmoidChunk)
    local outputGate   = nn.Narrow(2, 2 * layerSize + 1, layerSize) (sigmoidChunk)

    -- decode the write inputs
    local inTransform  = nn.Tanh()(nn.Narrow(2, 3 * layerSize + 1, layerSize)(allInputSums))

    -- perform the LSTM update
    local nextC = nn.CAddTable()({
      nn.CMulTable()({forgetGate, prevC}),
      nn.CMulTable()({inputGate, inTransform})
    })

    -- gated cells form the output
    local nextH = nn.CMulTable()({outputGate, nn.Tanh()(nextC)})

    table.insert(outputs, nextC)
    table.insert(outputs, nextH)
  end

  -- set up the decoder
  local topH = outputs[#outputs]
  if dropout > 0 then topH = nn.Dropout(dropout)(topH) end
  local proj = nn.Linear(layerSize, inputSize)(topH)
  local logSoftMax = nn.LogSoftMax()(proj)
  table.insert(outputs, logSoftMax)

  parent.__init(self, inputs, outputs)
end
