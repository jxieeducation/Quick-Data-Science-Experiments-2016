require 'torch'
require 'image'
require 'nn'
require 'optim'

noutputs = 10

nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

trainData = torch.load('data/processed_training_data.t7','ascii')

trainset = {}
function trainset:size() return trainData.size end
for i=1,trainData.size do
	trainset[i] = {trainData.data[i], trainData.labels[i]}
end

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,noutputs))

criterion = nn.MSECriterion()

-- example of using SGD
-- trainer = nn.StochasticGradient(model, criterion)
-- trainer.learningRate = 0.01
-- trainer:train(trainset)

-- example of training by hand
function gradUpdate(mlp, x, y, criterion, learningRate)
	local pred = mlp:forward(x)
	local err = criterion:forward(pred, y)
	print(err)
	local gradCriterion = criterion:backward(pred, y)
	-- print(gradCriterion:size())
	mlp:zeroGradParameters()
	mlp:backward(x, gradCriterion)
	mlp:updateParameters(learningRate)
end
for i=1,1000 do
	-- print(model:forward(trainData.data[1]))
	gradUpdate(model, trainData.data, trainData.labels, criterion, 0.01)
end

print(model:forward(trainData.data[1]))



