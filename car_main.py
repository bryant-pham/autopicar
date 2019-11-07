import numpy as np
import cv2
from driver import camera
from picar import back_wheels, front_wheels
import picar
import requests
import json
from time import sleep

HOST = '10.0.0.248'
PORT = '5000'
PATH = 'api/classify'

STATES = {
	0: 'straight',
	1: 'right', 
	3: 'veer_right',
	2: 'veer_left'
}
SLEEP_SECONDS = 2

current_state = 'straight'

def main():
	print('STARTING CAR')

	picar.setup()
	db_file = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
	fw = front_wheels.Front_Wheels(debug=True, db=db_file)
	bw = back_wheels.Back_Wheels(debug=True, db=db_file)
	cam = camera.Camera(debug=True, db=db_file)
	cam.ready()
	bw.ready()
	fw.ready()
	cv2Cam = cv2.VideoCapture(0)

	# bw.speed = 20
	turn_cam_down(cam)

	while True:
		print('TAKING PIC')
		np_img = take_picture(cv2Cam)

		print('CLASSIFYING PIC')
		prediction = classify(np_img)
		print('CLASSIFICATION: ' + prediction)

		if prediction == 'straight':
			if current_state == 'veer_left':
				turn_straight_from_left(fw) # Overcorrect due to faulty car
			else:
				turn_straight(fw)
		elif prediction == 'right':
			turn_right(fw)
		elif prediction == 'veer_right':
			veer_right(fw)
		elif prediction == 'veer_left':
			veer_left(fw)
		current_state = prediction

		print('SLEEPING')
		sleep_car(0.25)
		print('')
		print('')


def take_picture(cv2Cam):
	ret, frame = cv2Cam.read()
	return frame


def turn_cam_down(cam):
	print('----- TURNING CAM DOWN -----')
	cam.ready()
	cam.turn_down(32)


def turn_right(fw):
	print('----- TURNING RIGHT -----')
	fw.turn(132)


def turn_straight(fw):
	print('----- TURNING STRAIGHT -----')
	fw.turn_straight()


def turn_straight_from_left(fw):
	print('----- TURNING STRAIGHT WITH CORRECTION -----')
	fw.turn(100)


def veer_left(fw):
	print('----- VEERING LEFT -----')
	fw.turn(80)


def veer_right(fw):
	print('----- VEERING RIGHT -----')
	fw.turn(100)


def classify(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	content_type = 'image/jpeg'
	headers = {'content-type': content_type}
	url = build_url()
	print('POSTING URL: ' + url)
	_, img_encoded = cv2.imencode('.jpg', img)
	response = requests.post(url, data=img_encoded.tostring(), headers=headers)
	prediction = json.loads(response.text)
	return STATES[prediction]


def build_url():
	return 'http://' + '%s:%s/%s' % (HOST, PORT, PATH)


def sleep_car(secs):
	sleep(secs)

if __name__ == "__main__":
	main()


