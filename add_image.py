import sys
import requests
import numpy as np
import pickle

def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post('http://%s:8080/add/%s' % (sys.argv[2], sys.argv[3]), data=img)
    print response

if __name__ == '__main__':
	post_image(sys.argv[1])