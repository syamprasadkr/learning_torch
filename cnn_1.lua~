require 'nn'
require 'optim'
require 'cunn'

-- Fix seed for reproducibility
torch.manualSeed(1234)

-- Model begins
local model = nn.Sequential()
model:add(nn.SpatialConvolution(1, 64, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(64, 128, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(128, 256, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(256, 512, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.View(-1))
model:add(nn.Linear(2048, 1024))
model:add(nn.ReLU())
model:add(nn.Linear(1024, 5))
model:add(nn.LogSoftMax())
-- Model ends

-- Loss function
local criterion = nn.ClassNLLCriterion()

-- Loading data
train_set = torch.load("train_set.t7")
test_set = torch.load("test_set.t7")
classes = {"1", "2", "3", "4", "5"}


setmetatable(train_set, {
	__index = function(t, i)
				return {t.data[i], t.label[i]} 
	end});

train_set.data = train_set.data:double()

function train_set:size()
	return self.data:size(1)
end

X = torch.DoubleTensor(train_set.data)
Y = torch.DoubleTensor(train_set.label)
X = X[{{1, 1}}]
Y = Y[{{1, 1}}]
--X = train_set.data
--Y = train_set.label

-- Moving to GPU
model:cuda()
criterion:cuda()
X = X:cuda()
Y = Y:cuda()

-- Training
local params, gradParams = model:getParameters()
local optimState = {}

for epoch = 1, 50 do
	function feval(params)
		gradParams:zero()
		print("reached here1")
		local output = model:forward(X)
		print("reached here2")
		local loss = criterion:forward(output, Y)
		local dloss_doutput = criterion:backward(output, Y)
		model:backward(X, dloss_doutput)
		return loss, gradParams
	end
	print("Epoch: ", epoch)
	optim.sgd(feval, params, optimState)
end

