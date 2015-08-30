require 'torch'
require 'nn'

local OneHot, parent = torch.class('OneHot','nn.Module')

function OneHot:__init(outputSize)
    parent.__init(self)
    self.outputSize = outputSize
    -- create an identity matrix ???
    self.eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
    self.output:resize(input:size(1), self.outputSize):zero()
    if self.eye == nil then self.eye = torch.eye(self.outputSize) end
    -- cast the values from double to float
    self.eye = self.eye:float()
    local longInput = input:long()
    self.output:copy(self.eye:index(1, longInput))
    return self.output
end
