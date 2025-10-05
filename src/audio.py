import threading
import time
import re
import requests
import difflib
import os
from datetime import datetime

# Import speech recognition library
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except Exception:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False

# Speech recognizer class
class SpeechRecognizer:
    """
    Buffered background recognizer that accumulates chunks into sentences,
    uses regex + fuzzy matching to detect offensive/unsafe phrases and
    posts a rate-limited webhook alert.
    Public attributes:
      - speech_text: last completed sentence (str)
      - partial_text: last partial chunk (str)
      - offensive_detected: bool
      - help_detected: bool
    """
    def __init__(self, offensive_words=None, help_words=None, webhook_url=None,
                 phrase_time_limit=3, silence_flush_sec=0.8,
                 fuzzy_threshold=0.75, alert_cooldown=60):
        self.offensive_words = [w.strip().lower() for w in (offensive_words or []) if w.strip()]
        self.help_words = [w.strip().lower() for w in (help_words or []) if w.strip()]
        # compile regex patterns with word boundaries for exact/phrase matches
        self._regexes = [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in self.offensive_words]
        self._help_regexes = [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in self.help_words]
        self.webhook_url = webhook_url
        self.phrase_time_limit = phrase_time_limit
        self.silence_flush_sec = silence_flush_sec
        self.fuzzy_threshold = fuzzy_threshold
        self.alert_cooldown = alert_cooldown

        if SPEECH_RECOGNITION_AVAILABLE and sr is not None:
            self.r = sr.Recognizer()
            self.mic = sr.Microphone()
        else:
            self.r = None
            self.mic = None

        self.speech_text = ""
        self.partial_text = ""
        self.offensive_detected = False
        self.help_detected = False

        self._buffer = []
        self._lock = threading.Lock()
        self._flush_timer = None
        self._stop_listening = None
        self._last_alert_time = 0.0
        self._last_help_alert_time = 0.0
        
        # Initialize offensive words log file
        self.log_file = "offensive_words_log.txt"
        self._initialize_log_file()
    
    # Initialize log file with header
    def _initialize_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Timestamp,Offensive_Words_Detected,Sentence\n")

    # Send alert via webhook
    def _alert_webhook(self, sentence, event_type="offensive_speech"):
        now = time.time()
        if not self.webhook_url:
            return
        # Use different cooldowns for different alert types
        last_alert_time = self._last_alert_time if event_type == "offensive_speech" else self._last_help_alert_time
        if now - last_alert_time < self.alert_cooldown:
            return
        try:
            payload = {"event": event_type, "text": sentence, "timestamp": now}
            requests.post(self.webhook_url, json=payload, timeout=3)
            if event_type == "offensive_speech":
                self._last_alert_time = now
            else:
                self._last_help_alert_time = now
        except Exception:
            pass

    # Flush buffer and check for offensive content
    def _flush_buffer(self):
        with self._lock:
            if not self._buffer:
                return
            sentence = " ".join(self._buffer).strip()
            self._buffer = []
            self.partial_text = ""
            self.speech_text = sentence

            # Offensive detection: regex, substring/token, then fuzzy
            lower = sentence.lower()
            offensive_detected = False
            help_detected = False

            # 1) regex (word/phrase) exact matches for offensive words
            for rx in self._regexes:
                if rx.search(sentence):
                    offensive_detected = True
                    break

            # 2) regex (word/phrase) exact matches for help words
            for rx in self._help_regexes:
                if rx.search(sentence):
                    help_detected = True
                    break

            # 3) substring / token check for offensive words (catch single words inside sentences)
            if not offensive_detected and self.offensive_words:
                # token list
                tokens = re.findall(r"\w+", lower)
                for phrase in self.offensive_words:
                    if " " in phrase:
                        # multiword phrase -> substring search
                        if phrase in lower:
                            offensive_detected = True
                            break
                    else:
                        # single word -> token membership
                        if phrase in tokens:
                            offensive_detected = True
                            break

            # 4) substring / token check for help words (catch single words inside sentences)
            if not help_detected and self.help_words:
                # token list
                tokens = re.findall(r"\w+", lower)
                for phrase in self.help_words:
                    if " " in phrase:
                        # multiword phrase -> substring search
                        if phrase in lower:
                            help_detected = True
                            break
                    else:
                        # single word -> token membership
                        if phrase in tokens:
                            help_detected = True
                            break

            # 5) fuzzy fallback for offensive words
            if not offensive_detected and self.offensive_words:
                for phrase in self.offensive_words:
                    ratio = difflib.SequenceMatcher(None, phrase.lower(), lower).ratio()
                    if ratio >= self.fuzzy_threshold:
                        offensive_detected = True
                        break

            # 6) fuzzy fallback for help words
            if not help_detected and self.help_words:
                for phrase in self.help_words:
                    ratio = difflib.SequenceMatcher(None, phrase.lower(), lower).ratio()
                    if ratio >= self.fuzzy_threshold:
                        help_detected = True
                        break

            if offensive_detected and not self.offensive_detected:
                # only alert on new detection
                self._alert_webhook(sentence, "offensive_speech")
                # Log offensive words
                self._log_offensive_words(sentence)
            
            if help_detected and not self.help_detected:
                # only alert on new detection
                self._alert_webhook(sentence, "help_requested")

            self.offensive_detected = offensive_detected
            self.help_detected = help_detected
    
    # Log offensive words to file
    def _log_offensive_words(self, sentence):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Find which offensive words were detected
            detected_words = []
            lower_sentence = sentence.lower()
            for word in self.offensive_words:
                if word in lower_sentence:
                    detected_words.append(word)
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{';'.join(detected_words)},\"{sentence}\"\n")
        except Exception:
            pass

    # Schedule buffer flush
    def _schedule_flush(self):
        if self._flush_timer and self._flush_timer.is_alive():
            try:
                self._flush_timer.cancel()
            except Exception:
                pass
        self._flush_timer = threading.Timer(self.silence_flush_sec, self._flush_buffer)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    # Callback function for speech recognition
    def _callback(self, recognizer, audio):
        if not SPEECH_RECOGNITION_AVAILABLE or self.r is None:
            return
        try:
            text = recognizer.recognize_google(audio)
        except Exception:
            return
        with self._lock:
            self._buffer.append(text)
            self.partial_text = text
            # For more immediate feedback, we'll also update speech_text immediately for single words
            if len(self._buffer) == 1:
                self.speech_text = text
            # schedule flush after silence
            self._schedule_flush()

    # Start speech recognition
    def start(self):
        if not SPEECH_RECOGNITION_AVAILABLE or self.r is None or self.mic is None:
            print("SpeechRecognition not available; speech disabled.")
            return
        # ambient calibration
        try:
            with self.mic as source:
                self.r.adjust_for_ambient_noise(source, duration=1)
        except Exception:
            pass

        self._stop_listening = self.r.listen_in_background(
            self.mic, self._callback, phrase_time_limit=self.phrase_time_limit
        )

    # Stop speech recognition
    def stop(self):
        if self._stop_listening:
            try:
                self._stop_listening(wait_for_stop=False)
            except Exception:
                pass
            self._stop_listening = None
        if self._flush_timer and self._flush_timer.is_alive():
            try:
                self._flush_timer.cancel()
            except Exception:
                pass
        # flush remaining
        self._flush_buffer()