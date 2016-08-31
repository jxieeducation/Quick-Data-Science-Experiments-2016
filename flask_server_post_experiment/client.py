# http://stackoverflow.com/questions/37051441/large-fileabout-3gb-upload-with-urllib-sock-sendalldata-oserror

import urllib2
import sys
import requests
import pickle

def sendme(data, master='0.0.0.0:5000'):
    # headers = {'Content-Type': 'application/data'}
    # headers['Content-Length'] = sys.getsizeof(data)
    # print headers
    # request = urllib2.Request('http://%s/ping' % master, data, headers=headers)
    # return urllib2.urlopen(request).read()
    f = open('/tmp/rand', 'wb')
    pickle.dump(data, f)
    with open('/tmp/rand', 'rb') as g:
    	requests.post("http://"+master+"/ping", data=g)

    # response = urllib2.urlopen(urllib2.Request('http://%s/ping' % master, data, {'Content-Length': sys.getsizeof(data)}))

data = 'a' * 1000000 # this is 1mb
data = data * 3000
data += '????'

print "data size is %d" % sys.getsizeof(data)

sendme(data)
