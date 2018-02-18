import sys
import requests
import base64
import numpy

ids_raw = numpy.loadtxt('/home/lafonj/Documents/celeb_samples/Top1M_MidList.Name.tsv', delimiter='\t', dtype=str)
ids = {}
for _id, name_raw in ids_raw:
    name = name_raw.split('@')
    if len(name) == 2 and name[1] == 'en':
        ids[_id] = name[0].replace('"', '')
del ids_raw

def get_image(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    else:
        return base64.b64encode(resp.content)


def check_name(server, name):
    resp = requests.post('http://%s:5000/has_name' % server, json={'name': name})
    return resp.json()['has_name']

def add_image_name(url, server, name, no_duplicate=True):
    """ post image and return the response """
    
    #with open(img_file, 'rb') as f:
    #   img = base64.b64encode(f.read())

    if no_duplicate and check_name(server, name):
        print('Name already there. To store again call with `allow_duplicate=True')
        return

    img = get_image(url)
    if img is not None:
        data = {
            'img': img,
            'name': name,
            'filename': url
        }
        response = requests.post('https://%s:8080/add' % (server), json=data, verify=False)
        print response.text
        return
    else:
        print 'Image no longer exists in that url'
        return

def add_image_name2(path, server, name, no_duplicate=True):
    """ post image and return the response """
    
    with open(path, 'rb') as f:
       img = base64.b64encode(f.read())

    if no_duplicate and check_name(server, name):
        print('Name already there. To store again call with `allow_duplicate=True')
        return

    if img is not None:
        data = {
            'img': img,
            'name': name,
            'filename': url
        }
        response = requests.post('https://%s:8080/add' % (server), json=data, verify=False)
        print response.text
        return
    else:
        print 'Image no longer exists in that url'
        return


def add_image(url, _id, server='tx1', no_duplicate=True):
    address = server + '.cachengo.com'
    name = ids.get(_id)
    if name is None:
        print('Skipping because id not found. Probably because name was not available in English')
        return
    add_image_name(url, address, name, no_duplicate)



if __name__ == '__main__':
    # Example usage. Make sure to not overload the db!!!!
    samples = numpy.loadtxt('/home/lafonj/Documents/celeb_samples/MsCelebV1-Faces-Cropped.Samples.tsv', usecols=[0, 1, 2], dtype=str)
    for sample in samples:
        try:
            add_image(sample[2], sample[0])
        except:
            print('Failed. Probably got a weird url: %s' % sample[2])