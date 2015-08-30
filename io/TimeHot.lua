require 'torch'
require 'nn'

local TimeHot, parent = torch.class('TimeHot','nn.Module')

function TimeHot:__init(outputSize)
    parent.__init(self)
    self.outputSize = outputSize
end

function TimeHot:updateOutput(input)
    self.output:resize(input:size(1), self.outputSize):zero()
    self.output:copy(input)
    return self.output
end
