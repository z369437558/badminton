#coding=utf-8
import cv2
 
 
vc = cv2.VideoCapture('test.avi')  # 读入视频文件
c=1
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
 
timeF = 100  # 视频帧计数间隔频率
 
while rval:
    # 循环读取视频帧
    rval, frame = vc.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Display the resulting frame
    cv2.imshow('frame', frame)
 
    if (c % timeF == 0):
        # 每隔timeF帧进行存储操作
 
            cv2.imwrite('image' + str(c) + '.jpg', frame)  # 存储为图像
 
    c = c + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
vc.release()
cv2.destroyAllWindows()