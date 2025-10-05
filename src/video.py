import cv2

# Video capture class
class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Initialize webcam
        # Set optimized resolution for real-time processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Set frame rate
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Capture frame from webcam
    def get_frame(self):
        ret, frame = self.cap.read()  # Capture frame
        if not ret:
            return None
        return frame

    # Release the webcam
    def release(self):
        self.cap.release()

# Test function
def main():
    video_capture = VideoCapture()
    while True:
        frame = video_capture.get_frame()
        if frame is None:
            break
        cv2.imshow("Webcam Feed", frame)  # Display the frame
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()