-- combine all the parameters in a single flattened tensor
-- that is flatParameters and flatGradParameters
function utils.combineParams(...)

  local networks = {...}
  local parameters = {}
  local gradParameters = {}
  -- add the parameters
  for i = 1, #networks do
    local netParams, netGrads = networks[i]:parameters()

    if netParams then
      for _, p in pairs(netParams) do parameters[#parameters + 1] = p end
      for _, g in pairs(netGrads) do gradParameters[#gradParameters + 1] = g end
    end
  end

  local function storageInSet(set, storage)
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then return nil end
    local _, offset = unpack(storageAndOffset)
    return offset
  end

  --this function flattens arbitrary lists of parameters
  local function flatten(parameters)
    if not parameters or #parameters == 0 then
      return torch.Tensor()
    end
    local Tensor = parameters[1].new

    local storages = {}
    local nParameters = 0
    for k = 1, #parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = { storage, nParameters }
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k = 1, #parameters do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      parameters[k]:set(flatStorage,
      storageOffset + parameters[k]:storageOffset(),
      parameters[k]:size(),
      parameters[k]:stride())
      parameters[k]:zero()
    end

    local maskParameters = flatParameters:float():clone()
    local cumulativeSum = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumulativeSum[#cumulativeSum]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k = 1, #parameters do
      local offset = cumulativeSum[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
      parameters[k]:storageOffset() - offset,
      parameters[k]:size(),
      parameters[k]:stride())
    end

    for _, storageAndOffset in pairs(storages) do
      local k, v = unpack(storageAndOffset)
      flatParameters[{ { v + 1, v + k:size() } }]:copy(Tensor():set(k))
    end

    if cumulativeSum:sum() == 0 then
      flatUsedParameters:copy(flatParameters)
    else
      local counter = 0
      for k = 1, flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter + cumulativeSum[k]]
        end
      end
      assert(counter == nUsedParameters)
    end

    return flatUsedParameters
  end

  -- flatten parameters and gradients
  local flatParameters = flatten(parameters)
  local flatGradParameters = flatten(gradParameters)

  --return a flat vector tht contains all discrete parameters
  return flatParameters, flatGradParameters
end

function utils.clone(net, times)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, times do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i = 1, #paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

-- takes a list of tensors and returns a list of cloned tensors
function utils.cloneList(tensors, zeroToo)
    local out = {}
    for k, v in pairs(tensors) do
        out[k] = v:clone()
        if zeroToo then out[k]:zero() end
    end
    return out
end

function utils.merge(t1, t2)
  for k,v in pairs(t2) do t1[k] = v end
end

-- takes an input and convert it to CUDA or OpenCL based on opt parameter
function utils.checkArchitecture(what, opt)
  if opt.gpuId >= 0 then
    if opt.openCL then return what:cl()
    else return what:cuda() end
  end
  return what
end
