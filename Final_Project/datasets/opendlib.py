import dlib 
import cv2
import os

def face_path(path):

    file_paths=[] 
    file_path=os.listdir(path) 
    file_path.sort(key=lambda x:int(x[:-4]))

    for files in file_path:

        paths=path+'/'+files 
        file_paths.append(paths)

    return file_paths

def face_detction():

    file_path=face_path('test') 
    i=1 
    for f in file_path:

        img = cv2.imread(f, cv2.IMREAD_COLOR)

        b, g, r = cv2.split(img) 
        img2 = cv2.merge([r, g, b]) 
        detector = dlib.get_frontal_face_detector() 
        dets = detector(img, 1) 
        if len(dets)==0:
            print(i) 
        print("Number of faces detected: {}".format(len(dets)))

        for index, face in enumerate(dets):


            left = face.left() 
            top = face.top() 
            right = face.right() 
            bottom = face.bottom() # cv2.rectangle(img, (left, top), (right, bottom), (0, 255,0), 3)

            imgs=img[top:bottom,left:right] 
            cv2.imwrite('crop/'+str(i)+'.jpg', imgs)
            i = i + 1 


    cv2.destroyAllWindows()

face_detction()