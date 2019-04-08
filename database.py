import pyrebase
import random
from time import sleep

class Database:
	def __init__(self):
		print("initialize database")
		self.config = {
			"apiKey": "AIzaSyD945OUdvQFSh95UrcTOOhJTgXGkTloX9U",
			"authDomain": "iothack.firebaseapp.com",
			"databaseURL": "https://iothack.firebaseio.com",
			"storageBucket": "iothack.appspot.com"
		}
		self.firebase = pyrebase.initialize_app(self.config)
		self.db = self.firebase.database()

	def update_location(self, latitude, longitude, bot_id):
		data = {"latitude" : latitude,
				"longitude" : longitude
				}
		self.db.child("bots").child(bot_id).update(data)


db = Database()
db.update_location(32.866968, -117.215350, "bot1")

name = "bot"
latitude = 32.866968
longitude = -117.215350

sleep(5)
print("starting now....")
sleep(1)

for i in range(99):
	new_name = name + str(i)
	#latitude += i / 100
	#longitude += i / 100
	
	latitude = random.randint(-90,91)
	longitude = random.randint(-180, 181)
	db.update_location(latitude, longitude, new_name)

