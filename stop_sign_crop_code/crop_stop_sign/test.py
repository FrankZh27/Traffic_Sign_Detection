import cv2
import numpy as np
import glob
import os
def detect_haar(img, example=True):
    # Loads the classifier and reads the image. 
    classifier = cv2.CascadeClassifier("stopsign_classifier.xml")

    #img = cv2.imread(img)
    
    images = []
    t=0
    for filename in os.listdir(img):
        #print os.path.join(img,filename)
        img_temp = cv2.imread(os.path.join(img,filename))
        images.append(img_temp)
        if(t == 15):
            print filename
        t=t+1

    print len(images)
    
    for i in range(len(images)):
        print i
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        stop_signs = classifier.detectMultiScale(gray, 1.02, 10)
        if example == True:
            if len(stop_signs) == 0:
                continue
            for (x,y,w,h) in stop_signs:
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #print "hello"
                temp = images[i]
                image_new = temp[y:y+h, x:x+w]
                cv2.imwrite( "./cropped_image/"+str(i)+".jpg", image_new);
            '''
            c = cv2.imshow('imglll',image_new)
            c = cv2.waitKey(0)
            if 'q' == chr(c & 255):
                QuitProgram()        
            cv2.destroyAllWindows()   
            ''' 


if __name__ == '__main__':
	img = '../dataset/street_view/Stop_Sign_Dataset/'
	m = detect_haar(img, example=True)