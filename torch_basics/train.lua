require 'torch'
require 'image'
require 'nn'

noutputs = 10

nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

trainData = torch.load('data/processed_training_data.t7','ascii')

trainset = {}
function trainset:size() return trainData.size end -- 100 examples
for i=1,trainData.size do
	trainset[i] = {trainData.data[i], trainData.labels[i]}
end

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,noutputs))

-- model:add(nn.LogSoftMax())
-- criterion = nn.ClassNLLCriterion()
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer:train(trainset)

model:forward(trainData.data[1])

