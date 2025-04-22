# TwinCams: distance/length measurement using stereo cameras

The primary goal of this project was to develop a code capable of accurately calculating the distance between two points in 3D space. Additionally, the provided code includes features such as image capturing, and stereo camera calibration, all implemented with simple commands to facilitate easy use when needed, so there is no need to write separate OpenCV codes for these tasks :)


For calibration you need a checkerboard (also known as chessboard). You can use this [website](https://calib.io/pages/camera-calibration-pattern-generator) for pattern file, then print it out and place it on a board or a flat surface.
Note that holding the board during calibration is a bit tricky, and you may need to repeat the process a few times to get a good result. Also, use at least 8 images to ensure proper calibration.

This video is an example of measuring the length of an A4 sheet (Actual length is 297 mm):

Note that this was recorded when I used my old codes, so the code in background is different from what you see in this repo.

https://github.com/user-attachments/assets/5b1e3947-87b6-45f1-a299-38e99de085d6






