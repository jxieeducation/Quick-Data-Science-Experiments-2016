require 'torch'
require 'image'
require 'nn'

train_file = 'data/train_32x32.t7'
test_file = 'data/test_32x32.t7'
mysize = 1000

loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4)[{{1,mysize}, {}, {}, {}}],
   labels = loaded.y[1][{{1,mysize}}],
   size = mysize
}
one_hot_labels = torch.zeros(trainData.labels:size()[1], 10)
trainData.labels = one_hot_labels:scatter(2, trainData.labels:long():view(-1,1), 1)
trainData.data = trainData.data:float()

for i = 1, trainData.data:size()[1] do
	trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
channels = {'y','u','v'}
mean = {}
std = {}
for i,channel in ipairs(channels) do
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
for c in ipairs(channels) do
   for i = 1,trainData.data:size()[1] do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
end

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()
   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)
end

trainData.data = trainData.data:double()

torch.save('data/processed_training_data.t7', trainData, 'ascii')