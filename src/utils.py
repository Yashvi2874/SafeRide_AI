def check_offensive_words(transcription, offensive_words):
    """Check for offensive words in the transcribed text."""
    detected_words = [word for word in offensive_words if word in transcription.lower()]
    return detected_words

def load_offensive_words(file_path):
    """Load a list of offensive words from a text file."""
    with open(file_path, 'r') as file:
        offensive_words = file.read().splitlines()
    return offensive_words

def display_warning(detected_words):
    """Generate a warning message for detected offensive words."""
    if detected_words:
        return f"Warning: Offensive words detected - {', '.join(detected_words)}"
    return None

def process_frame(frame):
    """Process a video frame for display (e.g., resizing, filtering)."""
    # Example processing: convert to grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)