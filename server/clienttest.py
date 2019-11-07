import cv2
import numpy as numpy
import os
import requests
import json

# FOLDER='server/captures'
# FILE='capture-2019-11-0616-12-08.png'

# FOLDER='test/0-straight'
# FILE='test-blue-straight-1-000001.png'

FOLDER='testimgs'
FILE='train-capture-right-2019-11-06-18-08-53.png'

def main():
	content_type = 'image/jpeg'
	headers = {'content-type': content_type}
	path = os.getcwd()
	path = os.path.join(path, FOLDER, FILE)
	print(path)
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	_, img_encoded = cv2.imencode('.jpg', img)
	# send http request with image and receive response
	response = requests.post('http://10.0.0.248:5000/api/classify', data=img_encoded.tostring(), headers=headers)
	# decode response
	print(json.loads(response.text))


if __name__ == '__main__':
    main()