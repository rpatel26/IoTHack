#!/usr/bin/python3
from gpiozero import Servo
from time import sleep

myGPIO=13

# Min and Max pulse widths converted into milliseconds
# To increase range of movement:
#   increase maxPW from default of 2.0
#   decrease minPW from default of 1.0
# Change myCorrection using increments of 0.05 and
# check the value works with your servo.
myCorrection=.7
maxPW=(2.0+myCorrection)/1000
minPW=(1.0-myCorrection)/1000

myServo = Servo(myGPIO,min_pulse_width=minPW,max_pulse_width=maxPW)

print("Using GPIO17")
print("Max pulse width is set to 2.45 ms")
print("Min pulse width is set to 0.55 ms")

BIAS = 5
MAX_RANGE = 15
RANGE = 15

def rotate():
  print("Set value range -1.0 to +1.0")
  r = range(15, -16, -1)
  for value in r:
    value = float(value) / MAX_RANGE
    #if value > 1:
      #value = -1 + (value - 1)
    myServo.value = value
    print("Servo value set to " + str(value))
    sleep(0.05)

def rt():
  for i in range(-180, 180):
    myServo.value = i/180
    sleep(0.001)

'''
while True:
  #rotate(range(RANGE, -1, -1))
  #rotate(range(5, RANGE + 1))
  input()
  rt()
  #sleep(1)
''' 
