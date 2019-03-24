from socket import *  

class VideoSteaming:
    def __init__(self, addr, port):
        self.socket = socket(AF_INET,SOCK_DGRAM)  
        self.socket.connect((addr, port))  

        
    def send(self, data):
        self.socket.sendall(data) 

    def close(self):
        self.socket.close()

INTERVAL = 0.1

def main():
    import cv2
    from time import sleep
    ADDR = "192.168.43.192" # "10.25.252.77" #"10.25.60.12"
    PORT = 1024
    cap = cv2.VideoCapture(0) # ("/dev/video0") #(0)
    sender = VideoSteaming(ADDR, PORT)
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (320,240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, data = cv2.imencode(".jpg", gray)
        data = data.tostring()
        print(len(data))
        sender.send(data)
        sleep(INTERVAL)
    cap.release()
    sender.close()


if __name__ == "__main__":
    main()

