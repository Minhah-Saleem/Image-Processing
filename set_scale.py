import cv2
import numpy as np
import matplotlib.pyplot as plt

def set_scale(image,pixel_spacing,scale = 'mm',mode = 'axial'):
    #global pix_x,pix_y,pix_z
    (pix_z,pix_y,pix_x) =  pixel_spacing
    num = 0.2
    y = image.shape[0]
    x = image.shape[1]
    if mode == 'axial':
        tot_y = (pix_y*y)
        tot_x = (pix_x*x)
        f1 = round(tot_y/num)
        f2 = round(tot_x/num)
        image = cv2.resize(image.astype(np.uint8), (f2,f1), interpolation=cv2.INTER_CUBIC)
        big_spacing_y = round((round(tot_y/7)),-1)
        pix_big_spacing_y = round(big_spacing_y/num)
        big_spacing_x = round((round(tot_x/7)),-1)
        pix_big_spacing_x = round(big_spacing_x/num)
        
    elif mode =='sagittal':
        tot_y = (pix_z*y)
        tot_x = (pix_y*x)
        f1 = round(tot_y/num)
        f2 = round(tot_x/num)
        image = cv2.resize(image.astype(np.uint8), (f2,f1), interpolation=cv2.INTER_CUBIC)
        big_spacing_y = round((round(tot_y/7)),-1)
        pix_big_spacing_y = round(big_spacing_y/num)
        big_spacing_x = round((round(tot_x/7)),-1)
        pix_big_spacing_x = round(big_spacing_x/num)
    else:
        tot_y = (pix_z*y)
        tot_x = (pix_x*x)
        f1 = round(tot_y/num)
        f2 = round(tot_x/num)
        image = cv2.resize(image.astype(np.uint8), (f2,f1), interpolation=cv2.INTER_CUBIC)
        big_spacing_y = round((round(tot_y/7)),-1)
        pix_big_spacing_y = round(big_spacing_y/num)
        big_spacing_x = round((round(tot_x/7)),-1)
        pix_big_spacing_x = round(big_spacing_x/num)
        
    y = image.shape[0]
    x = image.shape[1]
    font = min((y*0.001),(x*0.001))
    print(font)
    st = round(max(x*0.01,2))
    bt = round(max(y*0.01,2))
    print(st,bt)
    thickness = round(font)
    tst = round(max(x*0.001,1))
    tbt = round(max(y*0.001,1))
    print(tst,tbt)
    small_spacing = int(big_spacing_y/10)
    pix_small_spacing = round(pix_big_spacing_y/2)
    pix_smallest_spacing = round(pix_big_spacing_y/10)
    #print('pix_big_sp: ',pix_big_spacing_y)
    #print('pix_sm_sp: ',pix_small_spacing)
    #print('pix_smlest_sp: ',pix_smallest_spacing)
    sc_img = np.zeros_like(image)
    sc_img = sc_img.astype(np.uint8)
    if len(sc_img.shape)<3:
        sc_img[:,st:st+tst] = 255
        sc_img[-(bt+tbt):-bt,:] = 255
    else:
        sc_img[:,st:st+tst,:] = 255
        sc_img[-(bt+tbt):-bt,:,:] = 255
    j = 0
    k = 0
    l = 0
    for i in range(y-1,-1,-1):
        try:
            trresb = k%pix_big_spacing_y
        except:
            trresb = 'error'
        try:
            trress = k%pix_smallest_spacing
        except:
            trress = 'error'
        if (not k==0) and (trresb==0):
            l=k
            j = j + big_spacing_y
            position = (round(st*4),i)
            if not k==pix_big_spacing_y:
                tx = str(j)
            else:
                tx = str(j) + scale#'mm'
            if len(sc_img.shape)<3:
                sc_img[i:i+tst,st:round(st*3)] = 255
                sc_img = cv2.putText(sc_img,tx,position,cv2.FONT_HERSHEY_SIMPLEX,font,(255),thickness) 
            else:
                sc_img[i:i+tst,st:round(st*3),:] = 255
                sc_img = cv2.putText(sc_img,tx,position,cv2.FONT_HERSHEY_SIMPLEX,font,(255,255,255),thickness) 
            
        elif (not k==0) and (k==int((l+l+pix_big_spacing_y)/2)):
            if len(sc_img.shape)<3:
                sc_img[i:i+tst,st:round(st*2.4)] = 255
            else:
                sc_img[i:i+tst,st:round(st*2.4),:] = 255
        
        elif (trress ==0):
            if len(sc_img.shape)<3:
                sc_img[i:i+tst,st:round(st*1.6)] = 255
            else:
                sc_img[i:i+tst,st:round(st*1.6),:] = 255
        k=k+1
    small_spacing = int(big_spacing_x/10)
    pix_small_spacing = int(pix_big_spacing_x/2)
    pix_smallest_spacing = int(pix_big_spacing_x/10)
    #print('pix_big_sp: ',pix_big_spacing_x)
    #print('pix_sm_sp: ',pix_small_spacing)
    #print('pix_smlest_sp: ',pix_smallest_spacing)
    
    
    #if len(sc_img.shape)<3:
    #    sc_img[:,st+thickness] = 255
    #    sc_img[-(st+thickness):-st,:] = 255
    #else:
    #    sc_img[:,st+thickness,:] = 255
    #    sc_img[-(st+thickness):-st,:,:] = 255
    
    j = 0
    k = 0
    for i in range(x):
        try:
            trresb = i%pix_big_spacing_x
        except:
            trresb = 'error'
        try:
            trress = i%pix_smallest_spacing
        except:
            trress = 'error'
        if (not i==0) and (trresb==0):
            k=i
            j = j + big_spacing_x
            position = (i,y-round(bt*4))
            if not i==pix_big_spacing_x:
                tx = str(j)
            else:
                tx = str(j) + scale# 'mm'
            if len(sc_img.shape)<3:
                sc_img[-round(bt*3):-bt,i:i+tbt] = 255
                sc_img = cv2.putText(sc_img,tx,position,cv2.FONT_HERSHEY_SIMPLEX,font,(255), thickness) 
            else:
                sc_img[-round(bt*3):-bt,i:i+tbt,:] = 255
                sc_img = cv2.putText(sc_img,tx,position,cv2.FONT_HERSHEY_SIMPLEX,font,(255,255,255), thickness) 
        elif (not i==0) and (i==int((k+k+pix_big_spacing_x)/2)):
            if len(sc_img.shape)<3:
                sc_img[-round(bt*2.4):-bt,i:i+tbt] = 255
            else:
                sc_img[-round(bt*2.4):-bt,i:i+tbt,:] = 255
        elif (trress==0):
            if len(sc_img.shape)<3:
                sc_img[-round(bt*1.6):-bt,i:i+tbt] = 255
            else:
                sc_img[-round(bt*1.6):-bt,i:i+tbt,:] = 255
    sc_img = sc_img.astype(np.uint8)
    image = np.where(sc_img==255,sc_img,image)
    return image     




##To call function
pix_x= 1
pix_y = 0.9
pix_z = 0.3
img = np.zeros((500,500), dtype = 'uint8')
image = set_scale(img,(pix_z,pix_y,pix_x),scale = 'cm', mode = 'axial')