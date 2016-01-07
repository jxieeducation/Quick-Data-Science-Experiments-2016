from sklearn.cross_validation import train_test_split
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data 
y = iris.target

train_file = open('iris_train', 'wb')
test_file = open('iris_test', 'wb')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1337)
def export_to_file(X_loc, y_loc, myfile):
	for i in range(X_loc.shape[0]):
		line = ""
		line += str(y_loc[i]) + " | "
		row = X_loc[i]
		line += "sental_length:" + str(row[0]) + " "
		line += "sental_width:" + str(row[1]) + " "
		line += "petal_length:" + str(row[2]) + " "
		line += "petal_width:" + str(row[3]) + " "
		line += "\n"
		myfile.write(line)

export_to_file(X_train, y_train, train_file)
export_to_file(X_test, y_test, test_file)
