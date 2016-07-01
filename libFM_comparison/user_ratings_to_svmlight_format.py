import sys
from random import shuffle

inputFile = sys.argv[1]
outputFile = sys.argv[2]

outs = []
for line in open(inputFile):
	line = line.strip().split()
	user = int(line[0])
	movie = int(line[1])
	rating = int(line[2])
	output = "%d %d:1 %d:1" % (rating, user, movie)
	outs += [output]

shuffle(outs)

outFileObj = open(outputFile, 'wb')
for line in outs:
	outFileObj.write(line)
	outFileObj.write("\n")
