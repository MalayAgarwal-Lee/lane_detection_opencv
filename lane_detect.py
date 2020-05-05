import cv2
import numpy as np
import skvideo.io


def canny_edge(frame):
    '''
    Function which performs Canny Edge Detection on an image

    Parameters:
        frame: np.array, the image to be processed

    Returns:
        np.array, a grayscale image with the edges in white
    '''

    # We do not need the color frame since we only need pixel intensity
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # This will give us the edges in the image
    # 50 and 155 represent minVal and maxVal
    # Which are used to determine which edges are continuous
    return cv2.Canny(grayscale_frame, 80, 240)


def segment(edge_frame):
    '''
    Function which finds the edges relevant to the lanes in the images

    Parameters:
        edge_frame: np.array, an image after Canny Edge Detection

    Returns:
        np.array, a grayscale images with the edges of the lanes in white
    '''

    # Get height of frame
    frame_height = edge_frame.shape[0]

    # This triangle corresponds to the area where the lane exists
    # This is hard-coded so the lane detection is not perfect
    triangle = np.array([[(50, frame_height),
                          (380, 290),
                          (800, frame_height)]])

    # This is currently a black image
    # The area bounded by the triangle above will be filled with white
    mask = np.zeros_like(edge_frame)
    mask.astype(np.float32)

    # This will make mask an array where the pixels
    # In the area bounded by the triangle above will be white
    cv2.fillPoly(mask, triangle, 255)

    # The input is a frame after canny detection
    # It has white pixels only for the edges
    # When we do a bit-wise and with the mask
    # Only the lane edges will be retained
    # Since all the area in the lane is white
    return cv2.bitwise_and(edge_frame, mask)


def get_line_coordinates(frame, slope, intercept):
    '''
    Function which computes the coordinates of two points for the lane line
    Relative to frame

    Parameters:
        frame: np.array, the image wrt which the coords are to be calculated
        slope: float, the slop of the lane line
        intercept: float, the y-intercept of the lane line

    Returns:
        int tuple, the coordinates of the two points in x1, y1, x2, y2 format
    '''

    y1 = frame.shape[0]

    y2 = int(y1 - 125)

    x1 = int((y1 - intercept) / slope)

    x2 = int((y2 - intercept) / slope)

    return x1, y1, x2, y2


def get_lane_lines(frame):
    '''
    Function which finds the left and right lane lines in an image

    Parameters:
        frame: np.array, the segmented image whose lane lines are to be found

    Returns:
        tuple of int tuples, the coordinates of the left and right lane lines
    '''

    # Perform a Hough transform on image to detect all lines
    lines = cv2.HoughLinesP(frame,
                            2,
                            np.pi / 180,
                            100,
                            np.array([]),
                            minLineLength=40,
                            maxLineGap=5)

    # Currently, lines has the shape (N, 1, 4), where N is the number of lines
    # Where 4 corresponds to x1, y1, x2, y2
    # We change it to (N, 4) (i.e 2D matrix)
    lines = lines.reshape((-1, 4))

    # Fit a straight line (deg = 1) using x1, y1, x2, y2 for each line in lines
    # This gives a (N, 2) array where the 2 corresponds to slop and y-intercept
    fitted_lines = [np.polyfit((x1, x2), (y1, y2), deg=1) for x1, y1, x2, y2 in lines]
    fitted_lines = np.array(fitted_lines)

    # When the slope of a line is < 0, it is on the left side of lane
    # So, we obtain all the lines of that nature by filtering
    # There are many such lines, so we take the average of slope and intercept
    left_avg = np.average(fitted_lines[fitted_lines[:, 0] < 0],
                          axis=0)

    # The remaining lines are to the right
    # The same operation is performed
    right_avg = np.average(fitted_lines[fitted_lines[:, 0] >= 0],
                           axis=0)

    # Calculate the x1, y1, x2, y2 coordinates for the left and right lines
    left_coors = get_line_coordinates(frame, *left_avg)
    right_coors = get_line_coordinates(frame, *right_avg)

    return left_coors, right_coors


def draw_lane_polygon(frame, coordinates):
    '''
    Function which draws the lane as a rectangle on the image

    Parameters:
        frame: np.array, the image on which the lane lines are to be drawn
        coordinates: tuple of int tuples, the coords for left, right lane lines
        thickness: int, the thickness with which the lane lines are to be drawn

    Returns:
        np.array, the image with the drawn lane lines
    '''

    # Create a mask with same dimensions as the image
    mask = np.zeros_like(frame)
    mask.astype(np.float32)

    # Get the coordinates of the lanes
    left_x1, left_y1, left_x2, left_y2 = coordinates[0]
    right_x1, right_y1, right_x2, right_y2 = coordinates[1]

    # Create an array representing a rectangle
    # Note: The ordering matters
    rectangle = np.array([[(left_x1, left_y1),
                           (right_x1, right_y1),
                           (right_x2, right_y2),
                           (left_x2, left_y2)]])

    # Fill the mask with the rectangle in green color
    cv2.fillPoly(mask, rectangle, (0, 225, 0))

    # Overlay the mask on the image with a 30% transparency
    return cv2.addWeighted(mask, 0.3, frame, 1, 1)


def main():
    # Open the video
    video = cv2.VideoCapture('input.mp4')

    # Initialize empty list swhich will hold all the frames with lanes drawn
    output_frames = []

    # Loop infinitely
    while True:
        # read() returns a boolean (flag) and frame (the frame)
        # The boolean is False if all frames from the video have been read
        flag, frame = video.read()

        # Check if all frames are read and break from loop if that is the case
        if not flag:
            break

        # Preserve a copy of the original frame
        original_frame = frame

        # Detect all edges in the image
        frame = canny_edge(frame)

        # Get edges relevant to the lanes only
        frame = segment(frame)

        # Get the coordinates for left and right lanes
        lane_coordinates = get_lane_lines(frame)

        # Draw the lanes
        lane_frame = draw_lane_polygon(original_frame, lane_coordinates)

        # Add drawn frame to the list of all frames
        output_frames.append(lane_frame)

        # Amount of time (ms) OpenCV will wait for before reading next frame
        cv2.waitKey(10)

    # Release the resources consumed by the video
    video.release()
    cv2.destroyAllWindows()

    # Write all the drawn frames into a video
    skvideo.io.vwrite('detected_lanes.mp4',
                      np.array(output_frames))


if __name__ == '__main__':
    main()
