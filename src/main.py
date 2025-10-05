import cv2
import threading
from gui import HybridAttentionSpeechGUI
from video import VideoCapture
from audio import SpeechRecognizer
from detection import detect_attention_and_drowsiness
import tkinter as tk
import os

# Main application class
class HybridAttentionSpeechApp:
    def __init__(self):
        self.video_stream = VideoCapture()
        
        # Help-related phrases that trigger alerts
        help_phrases = [
            "help", "police", "stop the car", "stop car", "emergency", 
            "danger", "call police", "get help", "need help", "help me",
            "pull over", "stop the vehicle", "emergency stop", "dangerous",
            "unsafe", "scared", "afraid", "threat", "assault", "harassment"
        ]
        
        # Initialize speech recognizer
        self.speech_recognizer = SpeechRecognizer(
            offensive_words=[
                "idiot", "stupid", "danger", "stop car",
                "kill", "rape", "murder", "assault", "threat",
                "shoot", "gun", "attack", "hurt"
            ],
            help_words=help_phrases,
            webhook_url=os.environ.get("OFFENSIVE_ALERT_WEBHOOK"),
            phrase_time_limit=3,
            silence_flush_sec=0.8,
            fuzzy_threshold=0.75,
            alert_cooldown=60
        )
        self.speech_recognizer.start()
        
        # Create GUI
        self.root = tk.Tk()
        self.gui = HybridAttentionSpeechGUI(self.root)
        
        # Control flag
        self.running = True
        
        # Track last displayed sentence to avoid duplicates
        self.last_displayed_sentence = ""
        
        # Track if we've shown alerts to prevent continuous display
        self.offensive_alert_shown = False
        self.help_alert_shown = False

    # Process video stream
    def start_video_stream(self):
        while self.running:
            frame = self.video_stream.get_frame()
            if frame is not None:
                # Process frame and get attention score, drowsiness alert, and yawning status
                result = detect_attention_and_drowsiness(frame, self.speech_recognizer)
                
                # Handle both old and new return value formats
                if len(result) == 4:
                    processed_frame, attention_score, drowsiness_alert, yawning_detected = result
                else:
                    # Old format (3 values)
                    processed_frame, attention_score, drowsiness_alert = result
                    yawning_detected = False
                
                # Update GUI with attention and drowsiness data
                self.gui.update_attention(attention_score)
                self.gui.update_drowsiness(drowsiness_alert)
                
                # Update video display in GUI
                self.gui.update_video(processed_frame)
                
                # Update GUI to refresh display
                self.root.update_idletasks()
                self.root.update()
            
            # Use a shorter wait time for more responsive UI
            # Check for ESC key press without displaying separate window
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                self.running = False
                break

    # Process audio transcription
    def start_audio_transcription(self):
        while self.running:
            # The speech recognizer updates its attributes automatically
            # We just need to read them and update the GUI
            speech_text = self.speech_recognizer.speech_text or self.speech_recognizer.partial_text or ""
            offensive_detected = self.speech_recognizer.offensive_detected
            help_detected = self.speech_recognizer.help_detected
            
            # Only update GUI if we have a new sentence
            if speech_text and speech_text != self.last_displayed_sentence:
                # Update GUI with transcription data
                self.gui.update_transcription(speech_text)
                self.last_displayed_sentence = speech_text
                
            # ALWAYS check for offensive language and help requests - FIXED LOGIC
            if offensive_detected:
                # Show offensive alert every time it's detected
                self.gui.show_offensive_warning("OFFENSIVE LANGUAGE DETECTED!")
                
            if help_detected:
                # Show help alert every time it's detected
                self.gui.show_help_warning("PASSENGER ASKING FOR HELP!")
            
            # Small delay to prevent excessive CPU usage
            # Use shorter delay for more responsive alerts
            cv2.waitKey(50)

    # Run the application
    def run(self):
        # Start video processing in a separate thread
        video_thread = threading.Thread(target=self.start_video_stream)
        video_thread.daemon = True
        video_thread.start()
        
        # Start audio processing in a separate thread
        audio_thread = threading.Thread(target=self.start_audio_transcription)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Run GUI in main thread
        self.root.mainloop()
        
        # Clean up when GUI is closed
        self.running = False
        self.speech_recognizer.stop()
        self.video_stream.release()

# Main entry point
if __name__ == "__main__":
    app = HybridAttentionSpeechApp()
    app.run()