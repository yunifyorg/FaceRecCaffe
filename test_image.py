import sys
import requests
import numpy as np
import pickle

def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    print(img)
    #response = requests.post('http://%s:8080/identify' % sys.argv[2], data=img)
    #print response.json()

if __name__ == '__main__':
	post_image(sys.argv[1])