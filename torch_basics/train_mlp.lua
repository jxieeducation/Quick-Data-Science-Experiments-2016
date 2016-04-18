require 'torch'
require 'image'
require 'nn'

noutputs = 10

nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height
nhiddens = ninputs / 2

trainData = torch.load('data/processed_training_data.t7','ascii')

trainset = {}
function trainset:size() return trainData.size end
for i=1,trainData.size do
	trainset[i] = {trainData.data[i], trainData.labels[i]}
end

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens,noutputs))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
-- note: NLL means that u dont return a vector, u just return index of a vector
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.05
trainer:train(trainset)

print(model:forward(trainData.data[1]))
