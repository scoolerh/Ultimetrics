import cv2
import os

def convert_video_to_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get frames from the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Replace 'input_video.mp4' and 'output_frames' with your file paths
    input_video_path = 'input_video.mp4'
    output_frames_folder = 'output_frames'

    convert_video_to_frames(input_video_path, output_frames_folder)
