import cv2
import threading
from gui import HybridAttentionSpeechGUI
from video import VideoCapture
from audio import SpeechRecognizer
from detection import detect_attention_and_drowsiness
import tkinter as tk
import os
import time

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
        
        # Driver scoring system
        self.attention_scores = []  # List to store attention scores
        self.ride_start_time = time.time()  # Track when the ride started
        self.ride_end_time = None  # Track when the ride ended
        self.offensive_language_count = 0  # Track offensive language detections
        
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
                
                # Store attention score for final calculation
                self.attention_scores.append(attention_score)
                
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
                # Increment offensive language counter
                self.offensive_language_count += 1
                
            if help_detected:
                # Show help alert every time it's detected
                self.gui.show_help_warning("PASSENGER ASKING FOR HELP!")
            
            # Small delay to prevent excessive CPU usage
            # Use shorter delay for more responsive alerts
            cv2.waitKey(50)

    # Calculate final driver score
    def calculate_final_score(self):
        if not self.attention_scores:
            return 0.0
        
        # Calculate average attention score
        avg_score = sum(self.attention_scores) / len(self.attention_scores)
        
        # Apply penalty for offensive language
        # Each offensive language detection reduces the score by 5 points
        offensive_penalty = min(self.offensive_language_count * 5, 30)  # Max penalty of 30 points
        adjusted_score = max(0, avg_score - offensive_penalty)
        
        # Calculate ride duration in minutes
        if self.ride_end_time:
            ride_duration = (self.ride_end_time - self.ride_start_time) / 60
        else:
            ride_duration = (time.time() - self.ride_start_time) / 60
        
        return round(adjusted_score, 1)

    # Get driver rating description
    def get_driver_rating(self, score):
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Average"
        elif score >= 50:
            return "Below Average"
        else:
            return "Poor"

    # Run the application
    def run(self):
        try:
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
        finally:
            # Clean up when GUI is closed
            self.running = False
            self.ride_end_time = time.time()  # Mark ride end time
            final_score = self.calculate_final_score()
            driver_rating = self.get_driver_rating(final_score)
            ride_duration = (self.ride_end_time - self.ride_start_time) / 60
            readings = len(self.attention_scores)
            
            # Show final score in GUI
            try:
                self.gui.show_final_score(final_score, driver_rating, ride_duration, readings, self.offensive_language_count)
            except:
                # Fallback to console if GUI is already closed
                pass
            
            # Print final score to console
            print(f"\n--- Ride Summary ---")
            print(f"Final Driver Attention Score: {final_score}%")
            print(f"Driver Rating: {driver_rating}")
            print(f"Ride Duration: {ride_duration:.1f} minutes")
            print(f"Total Attention Readings: {readings}")
            
            self.speech_recognizer.stop()
            self.video_stream.release()

# Main entry point
if __name__ == "__main__":
    app = HybridAttentionSpeechApp()
    app.run()