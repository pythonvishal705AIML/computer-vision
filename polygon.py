import cv2
import numpy as np

clicked_points = []
paused = False

def click_event(event, x, y, flags, param):
    global clicked_points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        update_display()
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Polygon Points:")
        for point in clicked_points:
            print(point)

def update_display():
    global frame
    display_frame = frame.copy()

s
    if len(clicked_points) > 1:
        for i in range(len(clicked_points)):
            cv2.line(display_frame, clicked_points[i], clicked_points[(i+1) % len(clicked_points)], (0, 255, 0), 2)

    for point in clicked_points:
        cv2.circle(display_frame, point, 3, (0, 0, 255), -1)

    cv2.imshow('Video', display_frame)

aa= r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/301"
video_source = r"rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/301"
aa=r"output.avi"
vid = cv2.VideoCapture(video_source)
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', click_event)

while True:
    if not paused:
        ret, frame = vid.read()
        if not ret or frame is None:
            print("End of video or error reading frame.")
            break
        update_display()

    key = cv2.waitKey(30) & 0xFF  
    if key == ord('q'):  
        break
    elif key == ord('p'): 
        paused = not paused
    elif key == ord('n') and paused: 
        ret, frame = vid.read()
        if ret and frame is not None:
            update_display()
        else:
            print("End of video or error reading next frame.")

vid.release()
cv2.destroyAllWindows()




2

# #
# import cv2
# import numpy as np
#
#
# clicked_points = []
# paused = False
#
#
#
# def click_event(event, x, y, flags, param):
#     global clicked_points, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#         update_display()
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         # Right click to print polygon points
#         print("Polygon Points:")
#         for point in clicked_points:
#             print(point)
#
#
# def update_display():
#     global frame
#     display_frame = frame.copy()
#
#     # Resize frame to a desired size (e.g., width=800)
#     desired_width = 800
#     height, width = display_frame.shape[:2]
#     aspect_ratio = width / height
#     new_height = int(desired_width / aspect_ratio)
#     resized_frame = cv2.resize(display_frame, (desired_width, new_height))
#
#    
#     if len(clicked_points) > 1:
#         for i in range(len(clicked_points)):
#             pt1 = (int(clicked_points[i][0] * desired_width / width), int(clicked_points[i][1] * new_height / height))
#             pt2 = (int(clicked_points[(i + 1) % len(clicked_points)][0] * desired_width / width),
#                    int(clicked_points[(i + 1) % len(clicked_points)][1] * new_height / height))
#             cv2.line(resized_frame, pt1, pt2, (0, 255, 0), 2)
#
#   
#     for point in clicked_points:
#         pt = (int(point[0] * desired_width / width), int(point[1] * new_height / height))
#         cv2.circle(resized_frame, pt, 3, (0, 0, 255), -1)
#
#     cv2.imshow('Video', resized_frame)
#
#
# 
# video_source = r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/401"
# vid = cv2.VideoCapture(video_source)
# cv2.namedWindow('Video')
# cv2.setMouseCallback('Video', click_event)
#
# while True:
#     if not paused:
#         ret, frame = vid.read()
#         if not ret or frame is None:
#             print("End of video or error reading frame.")
#             break
#         update_display()
#
#     key = cv2.waitKey(30) & 0xFF  # Wait for 30ms
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('p'):  # Press 'p' to pause/unpause
#         paused = not paused
#     elif key == ord('n') and paused:  # Press 'n' to go to next frame when paused
#         ret, frame = vid.read()
#         if ret and frame is not None:
#             update_display()
#         else:
#             print("End of video or error reading next frame.")
#
# vid.release()
# cv2.destroyAllWindows()




3
# import cv2
# import numpy as np

# # Create a list to store clicked points
# clicked_points = []
# paused = False

# # Define a callback function for mouse events
# def click_event(event, x, y, flags, param):
#     global clicked_points, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#         update_display()
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         # Right click to print polygon points
#         print("Polygon Points:")
#         for point in clicked_points:
#             print(point)

# def update_display():
#     global frame
#     display_frame = frame.copy()

#     # Resize frame to a desired size (e.g., width=800)
#     desired_width = 800
#     height, width = display_frame.shape[:2]
#     aspect_ratio = width / height
#     new_height = int(desired_width / aspect_ratio)
#     resized_frame = cv2.resize(display_frame, (desired_width, new_height))

#     # Draw all polygon lines
#     if len(clicked_points) > 1:
#         for i in range(len(clicked_points)):
#             pt1 = (int(clicked_points[i][0] * desired_width / width), int(clicked_points[i][1] * new_height / height))
#             pt2 = (int(clicked_points[(i + 1) % len(clicked_points)][0] * desired_width / width),
#                    int(clicked_points[(i + 1) % len(clicked_points)][1] * new_height / height))
#             cv2.line(resized_frame, pt1, pt2, (0, 255, 0), 2)

#     # Draw all points
#     for point in clicked_points:
#         pt = (int(point[0] * desired_width / width), int(point[1] * new_height / height))
#         cv2.circle(resized_frame, pt, 3, (0, 0, 255), -1)

#     # Draw the polygon
#     if len(clicked_points) > 1:
#         polygon_points = [(int(p[0] * desired_width / width), int(p[1] * new_height / height)) for p in clicked_points]
#         cv2.polylines(resized_frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

#     cv2.imshow('Video', resized_frame)

# # Set up video capture
# video_source = r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/401"
# vid = cv2.VideoCapture(video_source)
# cv2.namedWindow('Video')
# cv2.setMouseCallback('Video', click_event)

# while True:
#     if not paused:
#         ret, frame = vid.read()
#         if not ret or frame is None:
#             print("End of video or error reading frame.")
#             break
#         update_display()

#     key = cv2.waitKey(30) & 0xFF  # Wait for 30ms
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('p'):  # Press 'p' to pause/unpause
#         paused = not paused
#     elif key == ord('n') and paused:  # Press 'n' to go to next frame when paused
#         ret, frame = vid.read()
#         if ret and frame is not None:
#             update_display()
#         else:
#             print("End of video or error reading next frame.")

# vid.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # Create a list to store clicked points
# clicked_points = []
# paused = False
#
#
# # Define a callback function for mouse events
# def click_event(event, x, y, flags, param):
#     global clicked_points, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#         update_display()
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         # Right click to print polygon points
#         print("Polygon Points:")
#         for point in clicked_points:
#             print(point)
#
#
# def update_display():
#     global frame
#     display_frame = frame.copy()
#
#     # Draw all polygon lines
#     if len(clicked_points) > 1:
#         for i in range(len(clicked_points)):
#             cv2.line(display_frame, clicked_points[i], clicked_points[(i + 1) % len(clicked_points)], (0, 255, 0), 2)
#
#     # Draw all points
#     for point in clicked_points:
#         cv2.circle(display_frame, point, 3, (0, 0, 255), -1)
#
#     cv2.imshow('Video', display_frame)
#
#
# # Set up video capture
# video_source= r"rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/501"
# vid = cv2.VideoCapture(video_source)
# cv2.namedWindow('Video')
# cv2.setMouseCallback('Video', click_event)
#
# while True:
#     if not paused:
#         ret, frame = vid.read()
#         if not ret or frame is None:
#             print("End of video or error reading frame.")
#             break
#
#         # Resize frame to 1280x720
#         frame = cv2.resize(frame, (1280, 720))
#
#         update_display()
#
#     key = cv2.waitKey(30) & 0xFF  # Wait for 30ms
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('p'):  # Press 'p' to pause/unpause
#         paused = not paused
#     elif key == ord('n') and paused:  # Press 'n' to go to next frame when paused
#         ret, frame = vid.read()
#         if ret and frame is not None:
#             # Resize frame to 1280x720
#             frame = cv2.resize(frame, (1280, 720))
#             update_display()
#         else:
#             print("End of video or error reading next frame.")
#
# vid.release()
# cv2.destroyAllWindows()
