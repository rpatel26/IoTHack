import pynmea2
import serial
from database import Database

PORT = "/dev/ttyAMA0"
s = serial.Serial(PORT, baudrate = 9600, timeout = 0.5)
db = Database()

while True:
	try:
		data = s.readline()
		pass
		#print(data)
		data = data.decode("utf-8")		
		if data[0:6] == "$GPGGA":
			#print("matched")
			msg = pynmea2.parse(data)
			print("lat=%s, lon=%s" % (msg.lat, msg.lon))
			if msg.lat != "" and msg.lon != "":
				db.update_location(float(msg.lat), float(msg.lon), "bot_trash")
			#break
	except:
		pass
