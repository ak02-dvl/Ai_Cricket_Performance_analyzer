import cv2
import os


def save_uploaded_video(uploaded_file, upload_folder="uploads"):
    """
    Save uploaded Streamlit video to disk.
    Returns the saved file path.
    """

    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def get_video_info(video_path):
    """
    Extract video metadata.
    """

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": total_frames,
        "duration": duration
    }


def read_video(video_path):
    """
    Generator that yields frames one by one.
    """

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        yield frame

    cap.release()