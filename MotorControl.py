import RPi.GPIO as GPIO
from time import sleep

class MotorControl:
	def __init__(self, in1 = 16, in2 = 12, in3 = 19, in4 = 20, en1 = 26, en2 = 21):
		print("initializing motor control")
		self.in1 = in1
		self.in2 = in2
		self.en1 = en1
		self.in3 = in3
		self.in4 = in4
		self.en2 = en2
		self.setup_pins()

	def setup_pins(self):
		print("setting up pins")
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(self.in1,GPIO.OUT)
		GPIO.setup(self.in2,GPIO.OUT)
		GPIO.setup(self.en1,GPIO.OUT)	
		GPIO.output(self.in1,GPIO.LOW)
		GPIO.output(self.in2,GPIO.LOW)
		self.p1 = GPIO.PWM(self.en1,1000)
		self.p1.start(25)

		GPIO.setup(self.in3,GPIO.OUT)
		GPIO.setup(self.in4,GPIO.OUT)
		GPIO.setup(self.en2,GPIO.OUT)	
		GPIO.output(self.in3,GPIO.LOW)
		GPIO.output(self.in4,GPIO.LOW)
		self.p2 = GPIO.PWM(self.en2,1000)
		self.p2.start(25)

		self.p1.ChangeDutyCycle(75)
		self.p2.ChangeDutyCycle(75)


	def forward(self):
		print("forward")
		GPIO.output(self.in1, GPIO.HIGH)
		GPIO.output(self.in2, GPIO.LOW)	
		GPIO.output(self.in3, GPIO.HIGH)
		GPIO.output(self.in4, GPIO.LOW)

	def backward(self):
		print("backward")
		GPIO.output(self.in1, GPIO.LOW)
		GPIO.output(self.in2, GPIO.HIGH)	
		GPIO.output(self.in3, GPIO.LOW)
		GPIO.output(self.in4, GPIO.HIGH)


	def turn_right(self):
		print("turn right")
		GPIO.output(self.in1, GPIO.LOW)
		GPIO.output(self.in2, GPIO.LOW)
		GPIO.output(self.in3, GPIO.HIGH)
		GPIO.output(self.in4, GPIO.LOW)

	def turn_left(self):
		print("turn left")
		GPIO.output(self.in1, GPIO.HIGH)
		GPIO.output(self.in2, GPIO.LOW)
		GPIO.output(self.in3, GPIO.LOW)
		GPIO.output(self.in4, GPIO.LOW)
	
	def stop(self):
		print("stop")
		GPIO.output(self.in1, GPIO.LOW)
		GPIO.output(self.in2, GPIO.LOW)
		GPIO.output(self.in3, GPIO.LOW)
		GPIO.output(self.in4, GPIO.LOW)


mc = MotorControl()

while(1):
	mc.turn_left()
	sleep(2)
	mc.turn_right()
	sleep(2)
	mc.forward()
	sleep(2)
	mc.backward()
	sleep(2)
