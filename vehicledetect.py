import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import lanefind

#
# turn bbox into heatmap by adding 1 for each pixel inside the bbox
#
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
   
#
# apply a threshold to the heatmap to keep only "hot" areas
# 
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#
# this function will draw the bounding boxes that labels returns
#
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

#
# used for debuggin - dump the bboxes on the image
def draw_bboxes(img, bboxes,box_color=(255,0,0)):
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], box_color, 6)


#
# try to move all the color conversion into one function - to keep things straight
#
def convert_color(img, color_space='YCrCb'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    return feature_image
        
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
#def color_hist(img, nbins=32, bins_range=(0, 256)):
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image=convert_color(image,color_space)
        #print(feature_image.shape)
        
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

#
# we pickle the model so that we don't have to recompute it each time
# we first use the memory version, then load from disk, then compute (and save to disk)
#
dist_pickle = {}
def loadModelPickle():
    global dist_pickle
    if not dist_pickle:
        pickle_file_name='svc_pickle.p'
        if (not os.path.isfile(pickle_file_name)):
            fitModelAndPickle()
        dist_pickle = pickle.load( open(pickle_file_name, "rb" ) )
    return dist_pickle

#
# fit the model, and save out the result to our pickle for later use
#
def fitModelAndPickle():
    cars = glob.glob('vehicles/*/*.png')
    notcars = glob.glob('non-vehicles/*/*.png')
        
        
    # Reduce the sample size because
    #sample_size = 500
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    #0.9809
    if False:
        color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 16    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off
    #0.9907
    if True:
        color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 16    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off
    #0.9899
    if False:
        color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 32  # HOG orientations
        pix_per_cell = 16 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32) # Spatial binning dimensions
        hist_bins = 1024    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off

    
    print('About to extract features from',str(len(cars)),' cars')
    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    print('About to extract features from',str(len(notcars)),' notcars')
    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    print('About to run fit')
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy_score = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', round(accuracy_score, 4))
    # Check the prediction time for a single sample
    t=time.time()

    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hist_bins"] = hist_bins
    dist_pickle["color_space"] = color_space
    dist_pickle["accuracy_score"] = accuracy_score

    pickle.dump( dist_pickle, open( "svc_pickle.p", "wb" ) )

#
# This function takes a scale and a y range, and will stack up the data we need, and call predict on each square. 
#
# returns::
#    draw_img:  an image that has bounding boxes drawn on it (not used)
#    box_list:  a list of boxes that have someting of value
#    decision_list: the probability the box is useful
#    debug_boxes: used to show where the scan boxes are, for the writeup
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,color_space):
    box_list = []
    decision_list=[]
    debug_boxes=[]
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
        
        
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            #test_probabilities = svc.predict_proba(test_features)
            test_decision = svc.decision_function(test_features)

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)

            if test_prediction == 1:
                #print(test_decision)
                decision_list.append(test_decision)
                box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                
            debug_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

                
    return draw_img, box_list,decision_list,debug_boxes

# this could be a class - probably would be cleaner than globals
box_history = []
frame_number = 0
num_vehicles = 0
last_boxes = []

# clear the pipeline before running
def initializePipeline():
    global frame_number
    global box_history
    global num_vehicles
    global last_boxes
    frame_number = 0
    num_vehicles = 0

    box_history = []
    last_boxes = []

#
# this is the main pipeline
#
def getWindowParameters():
    find_car_params = [
     { "out_img": [] , "box_list":[], "decision_list":[], "debug_boxes":[], "ystart": 380, "yend": 660, "scale": 2.5},
     { "out_img": [] , "box_list":[], "decision_list":[], "debug_boxes":[], "ystart": 380, "yend": 656, "scale": 2},
     { "out_img": [] , "box_list":[], "decision_list":[], "debug_boxes":[], "ystart": 380, "yend": 656, "scale": 1.5},
     { "out_img": [] , "box_list":[], "decision_list":[], "debug_boxes":[], "ystart": 380, "yend": 550, "scale": 1.75},
     { "out_img": [] , "box_list":[], "decision_list":[], "debug_boxes":[], "ystart": 380, "yend": 530, "scale": 1.25}
    ]
    return find_car_params

def drawDebugBoxes(fcp,image,hilightcolor=(255,255,255),box_color=(255,0,0)):
        debug_box=fcp["debug_boxes"]
        debug_image=np.copy(image)
        draw_bboxes(debug_image,debug_box,box_color=box_color)
        cv2.rectangle(debug_image, debug_box[0][0], debug_box[0][1], hilightcolor, 6)
        return debug_image

#
#  We use both canny edge detection, and the decision property to remove things that don't look like cars
#  this improves the false positives by a big margin
def rejectFrames(image,box_list,dec_list,debug):
    box_list_trunc=[]
    i=0
    for bbox in box_list:
        cropped = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        canny = cv2.Canny(cropped,128,255)
        nz=np.count_nonzero(canny)
        px=canny.shape[0]*canny.shape[1]
        pc=nz/px*1.
        PCS=""
        if (pc>0.06): PCS="*"
        decs=""
        if (dec_list[i]>0.75): decs="*"
        if debug:
            print('decision:',dec_list[i],decs,' pc:',pc,PCS)
            decs=""
            plt.imshow(cropped)
            plt.show()
        if (pc>0.06 and dec_list[i]>0.55):
            box_list_trunc.append(bbox)
        i=i+1
    return box_list_trunc 

def carDetectionPipeline(image,debug=False):
    global frame_number
    global box_history
    global num_vehicles
    global last_boxes

    # load the model from the pickle (or cache)
    dist_pickle = loadModelPickle()
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]
    accuracy_score = dist_pickle["accuracy_score"]
    #print('Test Accuracy of SVC = ', round(accuracy_score, 4))
   
    # scan for cars with 5 different patterns
    # This is not optimized, probably overkill
    # 


    box_list=[]
    dec_list=[]
    debug_boxes=[]
    find_car_params = getWindowParameters()
    for fcp in find_car_params:
       fcp["out_img"], fcp["box_list"],fcp["decision_list"],fcp["debug_boxes"] = find_cars(image, fcp["ystart"],fcp["yend"],fcp["scale"], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,color_space)
       box_list = box_list + fcp["box_list"]
       debug_boxes = debug_boxes + fcp["debug_boxes"]
       dec_list = dec_list+ fcp["decision_list"]

    #
    #  We use both canny edge detection, and the decision property to remove things that don't look like cars
    #  this improves the false positives by a big margin
    box_list= rejectFrames(image,box_list,dec_list,False)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    
    heat_sum = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    labels = label(heatmap)
    #print(labels)
    #print(labels[1])
    num_vehicles=labels[1]
    
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    #
    #  store our bboxes and heatmaps between frames to improve the pipeline
    #
    box_history.append(box_list)
    if (len(box_history) > 5):
         del box_history[0]
    
    #print('box history len:',len(box_history))
    for i in range(len(box_history)):
        if (i==len(box_history)-1):
            heat_sum = add_heat(heat_sum,box_history[i]*3)
        else:
            heat_sum = add_heat(heat_sum,box_history[i])
        
    box_size = len(box_history)+3
    box_dif=8/box_size
    heat_sum=heat_sum*box_dif
    heat_mean = np.mean(heat_sum)
    heat_std = np.std(heat_sum)
    
    #heat_sum = apply_threshold(heat_sum,heat_mean+(heat_std))
    #heat_sum = apply_threshold(heat_sum,20)
    # heat_sum = apply_threshold(heat_sum,17)  -- AJS GOOD
    # TODO what should the threshold be?  can we calculate something better?
    heat_sum = apply_threshold(heat_sum,15)
    heat_sum = np.clip(heat_sum, 0, 255)


    # Find final boxes from heatmap using label function
    structure =  np.ones((3, 3))
    labels = label(heat_sum,structure=structure)
    draw_img_sum = draw_labeled_bboxes(np.copy(image), labels)
    last_boxes=labels[0]
    #draw_img_sum = cv2.addWeighted(draw_img_sum,0.5,img2,0.3,0)

    
    #
    # overlay heatmap or frame by frame on the top left
    #
    #heat_sum_small = cv2.resize(draw_img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
    
    #im = heat_sum_small
    #from PIL import Image
    #import matplotlib.cm as cm
    #im = Image.fromarray(np.uint8(cm.hot(heat_sum_small/255)*255))
    #im = np.asarray(im)
    #im = im[:,:,0:3]
    #print(im.shape)
    #draw_img_sum[0:im.shape[0], 0:im.shape[1] ] = im

    
    if debug:
        
        # Iterate through all detected cars
        if False:
         for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            print(bbox)
            cropped = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            print(cropped.shape)
            canny = cv2.Canny(cropped,128,255)
            plt.imshow(cropped)
            plt.show()
            plt.imshow(canny)
            plt.show()
            print(canny.shape)
            print(np.max(canny))
            print('----')
            nz=np.count_nonzero(canny)
            px=canny.shape[0]*canny.shape[1]
            print(nz)
            print(px)
            print(nz/px*1.)
            print('----')



        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(heat_sum, cmap='hot')
        plt.title('Heat Sum')
        fig.tight_layout()
        plt.show()
        plt.hist(heat_sum.ravel(), bins=256,  range=(1, 255),fc='k', ec='k')
        plt.show()
    frame_number=frame_number+1
    return draw_img_sum

from moviepy.editor import VideoFileClip
def combinedPipeline(image):
    image = lanefind.lanePipeline(image)
    image = carDetectionPipeline(image)
    return image

def processCombinedVideo(input_video,output):
    initializePipeline()
    clip1 = VideoFileClip(input_video)
    out_clip = clip1.fl_image(combinedPipeline)
    out_clip.write_videofile(output,audio=False)


from moviepy.editor import VideoFileClip
def clipVideo(input_video,output,start,end):
    initializePipeline()
    clip1 = VideoFileClip(input_video)
    clip1 = clip1.subclip(start,end)
    out_clip = clip1.fl_image(combinedPipeline)
    out_clip.write_videofile(output,audio=False)





if __name__ == '__main__':
    a = 1
    #processCombinedVideo('project_video.mp4','project_video_out.mp4')
    #processCombinedVideo('test_video.mp4','test_video_out.mp4')
    #processCombinedVideo('bayshore.mp4','bayshore_out.mp4')
    #processCombinedVideo('../CarND-Advanced-Lane-Lines/challenge_video.mp4','challenge_video_out.mp4')
    #clipVideo('test_video.mp4','test_video_clip.mp4',0,1)
    #clipVideo('project_video.mp4','project_video_clip.mp4',22,23)
    #clipVideo('project_video.mp4','project_video_clip.mp4',27,29)
    #clipVideo('project_video.mp4','project_video_clip.mp4',19,20)

