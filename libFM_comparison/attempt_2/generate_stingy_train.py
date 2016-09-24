# starving the train data, so any U/A combo in test does not exist in train

# on second thought, the dataset is already like this...., soooo this is useless

tr = open('data/ua.base', 'rb')
te = open('data/ua.test', 'rb')

test_combos = te.read().split('\n')
set_of_test_combos = set()
for line in test_combos:
	fields = line.split()
	if len(fields) > 2:
		set_of_test_combos.add((fields[0], fields[1]))

new_tr = open('no_combo_train.txt', 'wb')

train_examples = tr.read().split('\n')
for line in train_examples:
	fields = line.split()
	if len(fields) > 2 and (fields[0], fields[1]) not in set_of_test_combos:
		new_tr.write('%s\n' % line)
new_tr.close()
