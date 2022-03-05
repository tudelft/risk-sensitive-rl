import cv2
vidcap = cv2.VideoCapture('../experimentsCyberZoo/videos/cvar1_04.mp4')
success,image = vidcap.read()
count = 0
while success:
  if (count%2 == 0):
    cv2.imwrite("../experimentsCyberZoo/frames/sparse_max/frame%d.jpg" % (count+646), image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
  else:
    success,image = vidcap.read()
  count += 1
