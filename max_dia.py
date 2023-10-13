def max_dia(mask,p_x, p_y arrow = False):
    ''' This function finds the maximum diameter of objects in binary mask image. If arrow is True, an arrow will be drawn to show the direction of the max diameter. '''
    #mask = np.zeros((res0.shape[0],res0.shape[1]),dtype = 'uint8')
    #mask = np.where((res0[:,:,0]==255)&(res0[:,:,1]==0)&(res0[:,:,2]==0),1,mask)
    mask = np.where(mask>0,1,0)
    mask, num_lab = ndimage.label(mask)
    res0 = np.zeros((mask.shape[0],mask.shape[1],3))
    dist = []
    for i in range(num_lab):
        points = np.argwhere(mask==i+1)
        points = points[points[:,0].argsort()]
        distances = distance.cdist(points,points)
        if len(distances)>1:
            [p1,p2] = np.squeeze(np.where(distances == np.max(distances)))
            if isinstance(p1, np.ndarray): 
                p1 = p1[-1]
            if isinstance(p2, np.ndarray): 
                p2 = p2[-1]
            [ty,tx] = points[p1]
            [cy,cx] = points[p2]
            d = (math.sqrt((((ty-cy)*p_y)**2)+(((tx-cx)*p_x)**2)))
            dist.append(d)
	    if arrow==True:
            	res0 = cv2.arrowedLine(res0, (cx,cy), (tx,ty),(0,0,255),2)
        else:
            dist.append('none')
    return dist,res0
