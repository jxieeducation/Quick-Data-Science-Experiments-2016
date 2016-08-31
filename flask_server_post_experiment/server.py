from flask import Flask, request

app = Flask(__name__)

@app.route('/ping', methods=['GET', 'POST', 'PUT'])
def ping():
    print "new request!"
    try:
        data = request.data
        # print data[-1]
    except:
        print "NOOOO"
    return 'OK'

app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)
