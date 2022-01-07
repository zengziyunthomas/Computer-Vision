import cv2 as cv
import numpy as np 
import random
import math

# calculate 2-norm
norm=lambda x1,x2,x3,x4:math.sqrt((x1-x3)**2+(x2-x4)**2)

# show an image
def cv_show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# calculate the similarity of two points
def rho(vector1,vector2):
    dot=np.dot(vector1,vector2)
    norm=np.linalg.norm(vector1)*np.linalg.norm(vector2)
    return dot/norm

# the function to get keypoints and feature vectors of picture 1 and picture 2
def detectandcompute(picture1,picture2):
    img1_gray=cv.imread(picture1,0)
    img2_gray=cv.imread(picture2,0)
    sift=cv.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(img1_gray,None)
    kp2,des2=sift.detectAndCompute(img2_gray,None)
    return kp1,des1,kp2,des2

# the function to find out similar points in two pictures
def matches(des1,des2):
    match=[]
    for i in range(len(des1)):
        count=0
        # reject those (i,j) and (i,k) all have a high similarity
        buffer=[]         
        for j in range(len(des2)):
            if rho(des1[i],des2[j])>0.97:
                if count==0:
                    buffer=[rho(des1[i],des2[j]),i,j]
                    count+=1
                else:
                    buffer=[]
                    break
        if buffer!=[]:
            match.append(buffer) 
    match1=[]
    for i in range(len(des2)):
        count=0
        # reject those (i,j) and (i,k) all have a high similarity
        buffer=[]         
        for j in range(len(des1)):
            if rho(des2[i],des1[j])>0.97:
                if count==0:
                    buffer=[rho(des2[i],des1[j]),j,i]
                    count+=1
                else:
                    buffer=[]
                    break
        if buffer!=[]:
            match1.append(buffer) 
    print(len(match),len(match1))
    for i in range(len(match)):
        if match[i] not in match1:
            match[i]=0
    match=list(filter(lambda x: x!=0,match))
    # (optional) use when the picture is large, it can give those pairs with a higher similarity to get a high priority
    match.sort(key=lambda x:x,reverse=True) 
    print(len(match))
    return match

# the function to calculate the homography matrix by 4 pairs of points
def Homography(match,kp1,kp2,points):
    # get the location
    x1,y1=kp1[match[points[0]][1]].pt
    x11,y11=kp2[match[points[0]][2]].pt
    x2,y2=kp1[match[points[1]][1]].pt
    x21,y21=kp2[match[points[1]][2]].pt
    x3,y3=kp1[match[points[2]][1]].pt
    x31,y31=kp2[match[points[2]][2]].pt
    x4,y4=kp1[match[points[3]][1]].pt
    x41,y41=kp2[match[points[3]][2]].pt
    # Ah=0
    A=np.array([ [x1,y1,1,0,0,0,-x1*x11,-y1*x11],
        [0,0,0,x1,y1,1,-x1*y11,-y1*y11],
        [x2,y2,1,0,0,0,-x2*x21,-y2*x21],
        [0,0,0,x2,y2,1,-x2*y21,-y2*y21],
        [x3,y3,1,0,0,0,-x3*x31,-y3*x31],
        [0,0,0,x3,y3,1,-x3*y31,-y3*y31],
        [x4,y4,1,0,0,0,-x4*x41,-y4*x41],
        [0,0,0,x4,y4,1,-x4*y41,-y4*y41],
    ])
    b=np.array([[x11],[y11],[x21],[y21],[x31],[y31],[x41],[y41]])
    # check whether the 8*8 is a singular matrix
    if np.linalg.det(A)!=0:
        h=np.linalg.solve(A,b)
        # change 8*1 to 3*3
        H=[[float(h[0][0]),float(h[1][0]),float(h[2][0])],[float(h[3][0]),float(h[4][0]),float(h[5][0])],[float(h[6][0]),float(h[7][0]),1.0]]
        return H
    else:
        return 0

# the function to choose a better homography matrix
def RANSAC(match,kp1,des1,kp2,des2):
    N=99999999
    sample_count=0
    H=[]
    while N>sample_count:
        # random 4 points and get a valid homography
        n=0
        while n==0:
            points=[]
            for i in range(4):
                points.append(random.randint(0,len(match)-1))
            H=Homography(match,kp1,kp2,points)
            if H!=0:
                break
        # use the homography to get the epsilon
        count=0
        epsilon=0
        for i in match:
            x,y=kp1[i[1]].pt
            origin=np.array([[x],[y],[1]])
            after=np.dot(H,origin)
            after[0][0]=after[0][0]/after[2][0]
            after[1][0]=after[1][0]/after[2][0]
            after[2][0]=1
            error=norm(after[0][0],after[1][0],kp2[i[2]].pt[0],kp2[i[2]].pt[1])
            if error<10:
                count+=1
        if epsilon<count/len(match):
            epsilon=count/len(match)
            N=math.log(0.01)/math.log(1-(epsilon)**4)    
        sample_count+=1
    return H

# the body of stitching two pictures
def stitch(picture1,picture2):
    # detect and compute the keypoit and feature vector of picture 1 and picture2
    kp1,des1,kp2,des2=detectandcompute(picture1,picture2)
    print(len(des1),len(des2))
    img1_color=cv.imread(picture1)
    img2_color=cv.imread(picture2)
    # get the match list of picture 1 and picture2
    match=matches(des1,des2)
    print("Success!!")
    # use the RANSAC algorithm to calculate a good homography matrix
    H=RANSAC(match,kp1,des1,kp2,des2)
    H=np.array(H)
    print(H)
    # use the homography matrix to tramform picture 1
    result=cv.warpPerspective(img1_color,H,(img1_color.shape[1]+img2_color.shape[1],img1_color.shape[0]))
    # put them together
    result[0:img2_color.shape[0],0:img2_color.shape[1]]=img2_color
    return result

# resize the size of the picture 
def downsampling(picture):
    p=cv.imread(picture)
    p=cv.resize(p,(480,640))
    cv.imwrite(picture,p)

if __name__=="__main__":
    # four picture here
    all_picture=["p1.jpg","p2.jpg","p3.jpg","p4.jpg"]
    # all_picture=["f1.jpg","f2.jpg"]
    # for a quick computing, please resize the picture
    # for i in all_picture:
    #     downsampling(i)
    result1=stitch(all_picture[0],all_picture[1]) 
    cv.imwrite("result1.jpg",result1)
    result2=stitch("result1.jpg",all_picture[2])
    cv.imwrite("result2.jpg",result2)
    result=stitch("result2.jpg",all_picture[3])
    cv.imwrite("result.jpg",result)
    # result=stitch("f2.jpg","f1.jpg")
    cv_show("image",result)
    
    
    
        



