require 'torch'
require 'image'
require 'nn'

noutputs = 10

nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height
nhiddens = ninputs / 2

nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

trainData = torch.load('data/processed_training_data.t7','ascii')

trainset = {}
function trainset:size() return trainData.size end
for i=1,trainData.size do
	trainset[i] = {trainData.data[i], trainData.labels[i]}
end

model = nn.Sequential()

model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))


model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
-- note: NLL means that u dont return a vector, u just return index of a vector
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.05
trainer:train(trainset)

print(model:forward(trainData.data[1]))
