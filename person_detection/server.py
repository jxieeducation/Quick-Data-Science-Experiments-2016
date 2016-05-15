from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      newFileName = "test." + f.filename.rsplit(".")[-1]
      f.save(newFileName) # we want to keep the file extension
      # image recognition stuff here
      return 'file uploaded successfully'

if __name__ == '__main__':
   app.run(debug = True)
