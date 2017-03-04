from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pickle
from collections import deque
import os.path
import glob



class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.badframenum = 0
        self.badframe = False
        self.badframeratio = False

    #
    # validate the new values, before we put them into the line
    #
    # return false if we have too many bad frames
    def validate_and_update(self,allx,ally,fitx,curverad,position):

        self.badframe = False

        #
        #  validate before changing
        #
        if self.allx is not None:
        
            # check to see if the curverad changes too much
            #print("=====")
            percentage = abs(((self.radius_of_curvature - curverad)/curverad))
            #print(self.radius_of_curvature,' ', curverad,' ',percentage)
         
            if ( (percentage > 30) and self.badframenum < 3):
               #print("TOO FAR?")
               self.badframe = True
               self.badframeratio = percentage
               self.badframenum = self.badframenum + 1
               return
       
            self.badframenum=0 
        #
        #  update  
        #
        self.allx = allx
        self.ally = ally

        self.current_fit = fitx #np.polyfit(self.allx, self.ally, 2)


        self.recent_xfitted.append(self.current_fit)
        # if we are larger than our averaging window, drop one
        if (len(self.recent_xfitted) > 5):
           self.recent_xfitted = self.recent_xfitted[1:]
 
        # best_fit is the average of the recent_xfitted 
        if (len(self.recent_xfitted) > 1):
            #self.best_fit = np.mean(self.recent_xfitted)
            self.best_fit = (self.best_fit * 4 + self.current_fit) / 5
        else:
            self.best_fit = self.current_fit
 
        self.radius_of_curvature = curverad
        return


def createCameraDistortionPickle(pickle_file_name):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print(fname)
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            #print(fname+' found')
            objpoints.append(objp)
            imgpoints.append(corners)

        else:
            print(fname+' not found')

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    if not os.path.exists('camera_cal_output'):
        os.makedirs('camera_cal_output')

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal_output/dist_pickle.p", "wb" ) )

#
#  This function for our pipeline removes the camera distortion
#  we need to load the camera information if it isn't loaded already

dist2_pickle = {}

def correctCameraDistortion(image):
    global dist2_pickle
    if not dist2_pickle:
        # load the pickle
        pickle_file_name='camera_cal_output/dist_pickle.p'
        if (not os.path.isfile(pickle_file_name)):
            createCameraDistortionPickle(pickle_file_name)
        dist2_pickle=pickle.load(open( 'camera_cal_output/dist_pickle.p', "rb" ) )
    mtx=dist2_pickle["mtx"]
    dist=dist2_pickle["dist"]
    img_size = (image.shape[1], image.shape[0])
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    return dst

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
# http://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv
def yellowRGBToBinary(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_y = np.array([20, 100, 100])
    upper_y = np.array([30, 255, 255])
    binary = cv2.inRange(hsv, lower_y, upper_y)
    return binary

def color_gradient_pipeline(img, s_thresh=(200, 255), sx_thresh=(30, 100)):
    img = np.copy(img)

    yellow_mask = yellowRGBToBinary(img)


    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    binary_output =  np.zeros_like(s_binary)
    #binary_output[(s_binary==1) & (sxbinary==1)] = 1
    binary_output[((s_binary == 1) | (sxbinary == 1) | (yellow_mask==1))] = 1
    return binary_output
    #return color_binary
    
def projectImageToBird(image):
    src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (image.shape[1], image.shape[0])
    binary_warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return binary_warped

def projectBirdToImage(image):
    src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (image.shape[1], image.shape[0])
    binary_warped = cv2.warpPerspective(image, Minv, img_size, flags=cv2.INTER_LINEAR)
    return binary_warped



def findLanesByHistogram(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    binary_warped = np.uint8(binary_warped*255)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return leftx,lefty,rightx,righty




def findCurvature(leftx,rightx,ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)


    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad,right_curverad
    
def findPositionInLane(centerx, leftx,rightx):
    center = (rightx + leftx) / 2
    position = (centerx - center[719]) * 3.7 / 700
    return position


    
def drawOutput(orig,left_fitx,right_fitx,ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(orig[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp=projectBirdToImage(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(orig, 1, newwarp, 0.3, 0)
    return result
   


left_line = Line() 
right_line = Line()

 
def lanePipeline(image):
    global left_line
    global right_line

    # correct the camera distortion 
    orig_img = correctCameraDistortion(image)

    # threshold the image and try to pull just the lines out
    img = color_gradient_pipeline(orig_img, (120, 254), (30, 100))
    # switch to birds eye projection
    warped=projectImageToBird(img)
    # fix the image to be 0-255 instead of 0-1
    binary_warped = np.uint8(warped*255)

    #
    #  Optimize the histogram if we already have two good lines
    #
    #if (left_line is not None and right_line is not None):
      

    # find the lanes using the historgram method
    leftx,lefty,rightx,righty = findLanesByHistogram(binary_warped)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    line_is_parallel=False
    if (np.abs(left_fit[0]-right_fit[0]) < 0.0004 and np.abs(left_fit[1]-right_fit[1])<0.6 ):
      line_is_parallel=True

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # find the curvature of the left and right lane marker
    left_curverad,right_curverad = findCurvature(left_fitx,right_fitx,ploty)
    # find where we are in the lane
    position = findPositionInLane((img.shape[1]/2),left_fitx,right_fitx)

    # let's skip when they aren't parallel -- we should probably only do this for 5 frames (TODO)
    if (left_line.best_fit is None or right_line.best_fit is None or line_is_parallel):
       left_line.validate_and_update(leftx,lefty,left_fitx,left_curverad,position)
       right_line.validate_and_update(rightx,righty,right_fitx,right_curverad,position)
       

    #
    # draw the lines onto the image
    #
    result=drawOutput(orig_img,left_line.best_fit,right_line.best_fit,ploty)

    #
    #  draw the radius and position on image
    #
    if (left_line.badframe):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'bad frame left {:.2f}'.format(left_line.badframeratio)
        cv2.putText(result,text,(10,80), font, 1,(255,255,255),2)
    if (right_line.badframe):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'bad frame right {:.2f}'.format(right_line.badframeratio)
        cv2.putText(result,text,(10,105), font, 1,(255,255,255),2)
    if (line_is_parallel is False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'not parallel'
        cv2.putText(result,text,(10,130), font, 1,(255,255,255),2)

    text = 'Radius of Curvature: {:.0f}m'.format((left_line.radius_of_curvature+right_line.radius_of_curvature)/2.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,text,(10,25), font, 1,(255,255,255),2)

    leftright='right'
    if (position<0):
      position=position*-1
      leftright='left'

    text = 'Vehicle is {:.2f}m {:s} of center'.format(position,leftright)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,text,(10,50), font, 1,(255,255,255),2)
    return result

def processVideo(input_video,output):
  clip1 = VideoFileClip(input_video)
  out_clip = clip1.fl_image(lanePipeline)
  out_clip.write_videofile(output,audio=False)


if __name__ == '__main__':
    processVideo('project_video.mp4','project_video_out.mp4')
    processVideo('challenge_video.mp4','challenge_video_out.mp4')
    processVideo('harder_challenge_video.mp4','harder_challenge_video_out.mp4')
