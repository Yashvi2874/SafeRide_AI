import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from PIL import Image, ImageTk
import cv2

# Main GUI class
class HybridAttentionSpeechGUI:
    def __init__(self, master):
        self.master = master
        master.title("🚗 SafeRide AI - Driver Monitoring System")
        master.geometry("1200x700")
        master.minsize(800, 500)  # Minimum window size
        master.resizable(True, True)
        master.configure(bg="#121212")  # Dark background

        # Configure styles for dark theme
        self.master.option_add("*Font", "Helvetica 10")
        
        # Header frame
        self.header_frame = tk.Frame(master, bg="#121212", pady=10)
        self.header_frame.pack(fill=tk.X)
        self.header_frame.columnconfigure(0, weight=1)
        
        # Title
        self.title_label = tk.Label(
            self.header_frame, 
            text="🚗 SafeRide AI", 
            font=("Helvetica", 20, "bold"), 
            fg="#4FC3F7", 
            bg="#121212"
        )
        self.title_label.pack()
        
        # Subtitle
        self.subtitle_label = tk.Label(
            self.header_frame, 
            text="Advanced Driver Monitoring System", 
            font=("Helvetica", 12), 
            fg="#B0B0B0", 
            bg="#121212"
        )
        self.subtitle_label.pack()

        # Main content frame with two columns
        self.content_frame = tk.Frame(master, bg="#121212")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid weights for responsive resizing - EQUAL weights for both columns
        self.content_frame.columnconfigure(0, weight=1)  # Video column
        self.content_frame.columnconfigure(1, weight=1)  # Info column
        self.content_frame.rowconfigure(0, weight=1)
        
        # Left column - Video monitoring
        self.video_column = tk.Frame(self.content_frame, bg="#121212")
        self.video_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_column.columnconfigure(0, weight=1)
        self.video_column.rowconfigure(1, weight=1)
        
        # Video frame
        self.video_frame = tk.Frame(self.video_column, bg="#1E1E1E", relief=tk.RAISED, bd=1)
        self.video_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(1, weight=1)
        
        self.video_title = tk.Label(
            self.video_frame, 
            text="📹 LIVE DRIVER MONITORING", 
            font=("Helvetica", 14, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E"
        )
        self.video_title.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        
        # Video display label with centered content
        self.video_label = tk.Label(self.video_frame, bg="#000000", anchor="center")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Right column - Information and alerts
        self.info_column = tk.Frame(self.content_frame, bg="#121212")
        self.info_column.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.info_column.columnconfigure(0, weight=1)
        
        # Attention level with progress bar
        self.attention_frame = tk.Frame(self.info_column, bg="#1E1E1E", relief=tk.RAISED, bd=1)
        self.attention_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.attention_frame.columnconfigure(0, weight=1)
        
        self.attention_title = tk.Label(
            self.attention_frame, 
            text="ATTENTION LEVEL", 
            font=("Helvetica", 14, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E"
        )
        self.attention_title.pack(pady=(10, 5))
        
        self.attention_label = tk.Label(
            self.attention_frame, 
            text="0%", 
            font=("Helvetica", 28, "bold"), 
            fg="#FFFFFF", 
            bg="#1E1E1E"
        )
        self.attention_label.pack(pady=5)
        
        # Progress bar for attention level
        self.progress_frame = tk.Frame(self.attention_frame, bg="#1E1E1E")
        self.progress_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.progress_frame.columnconfigure(0, weight=1)
        
        self.progress_canvas = tk.Canvas(self.progress_frame, height=25, bg="#333333", highlightthickness=0)
        self.progress_canvas.grid(row=0, column=0, sticky="ew", padx=5)
        
        # Status indicators frame
        self.status_frame = tk.Frame(self.info_column, bg="#1E1E1E", relief=tk.RAISED, bd=1)
        self.status_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.status_frame.columnconfigure(0, weight=1)
        
        self.status_title = tk.Label(
            self.status_frame, 
            text="DRIVER STATUS", 
            font=("Helvetica", 14, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E"
        )
        self.status_title.pack(pady=(10, 5))
        
        self.drowsiness_label = tk.Label(
            self.status_frame, 
            text="Normal", 
            font=("Helvetica", 16, "bold"), 
            fg="#4CAF50", 
            bg="#1E1E1E"
        )
        self.drowsiness_label.pack(pady=5)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="", 
            font=("Helvetica", 12), 
            fg="#B0B0B0", 
            bg="#1E1E1E"
        )
        self.status_label.pack(pady=(0, 10))

        # Speech transcription section
        self.transcription_frame = tk.Frame(self.info_column, bg="#1E1E1E", relief=tk.RAISED, bd=1)
        self.transcription_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.transcription_frame.columnconfigure(0, weight=1)
        self.transcription_frame.rowconfigure(1, weight=0)  # Don't expand automatically
        
        self.transcription_title = tk.Label(
            self.transcription_frame, 
            text="SPEECH TRANSCRIPTION", 
            font=("Helvetica", 12, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E"
        )
        self.transcription_title.pack(pady=(10, 5))

        # Scrollable text area for transcription with fixed height
        self.transcription_text_frame = tk.Frame(self.transcription_frame, bg="#1E1E1E")
        self.transcription_text_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))
        self.transcription_text_frame.columnconfigure(0, weight=1)
        self.transcription_text_frame.rowconfigure(0, weight=1)
        
        self.transcription_text = scrolledtext.ScrolledText(
            self.transcription_text_frame, 
            wrap=tk.WORD, 
            font=("Helvetica", 10),
            bg="#2D2D2D",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            relief=tk.FLAT,
            bd=2,
            height=8  # Fixed height to minimize space
        )
        self.transcription_text.grid(row=0, column=0, sticky="nsew")

        # Alert frames - MAKE THEM VISIBLE BY DEFAULT
        self.alerts_frame = tk.Frame(self.info_column, bg="#121212")
        self.alerts_frame.grid(row=3, column=0, sticky="ew")
        self.alerts_frame.columnconfigure(0, weight=1)
        
        # Offensive warning label - VISIBLE BY DEFAULT
        self.offensive_warning_label = tk.Label(
            self.alerts_frame, 
            text="Monitoring for offensive language", 
            font=("Helvetica", 16, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E",
            relief=tk.RAISED, 
            bd=3,
            padx=20,
            pady=15
        )
        self.offensive_warning_label.grid(row=0, column=0, sticky="ew", pady=5)
        
        # Help warning label
        self.help_warning_label = tk.Label(
            self.alerts_frame, 
            text="Monitoring for help requests", 
            font=("Helvetica", 16, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E",
            relief=tk.RAISED, 
            bd=3,
            padx=20,
            pady=15
        )
        self.help_warning_label.grid(row=1, column=0, sticky="ew", pady=5)
        
        # Initialize variables for video display
        self.photo_image = None
        
        # Initialize last attention value
        self.last_attention = 0
        
        # Bind resize event to update progress bar
        self.master.bind('<Configure>', self.on_resize)

    # Update attention display
    def update_attention(self, attention):
        # Store last attention value for resize updates
        self.last_attention = attention
        
        # Update attention percentage
        self.attention_label.config(text=f"{attention:.1f}%")
        
        # Update attention color based on level
        if attention > 70:
            self.attention_label.config(fg="#4CAF50")  # Green
        elif attention > 40:
            self.attention_label.config(fg="#FFC107")  # Yellow
        else:
            self.attention_label.config(fg="#FF5252")  # Red
        
        # Update progress bar
        self._update_attention_progress(attention)
    
    # Update attention progress bar
    def _update_attention_progress(self, attention):
        # Update progress bar
        self.progress_canvas.delete("all")
        self.progress_canvas.update_idletasks()  # Ensure canvas is updated
        width = self.progress_canvas.winfo_width()
        height = self.progress_canvas.winfo_height()
        if width <= 1:  # Handle initial case when width is not set
            width = max(300, self.progress_canvas.winfo_reqwidth())
        if height <= 1:  # Handle initial case when height is not set
            height = 25
        
        # Draw background
        self.progress_canvas.create_rectangle(0, 0, width, height, fill="#333333", outline="")
        
        # Draw progress
        progress_width = int((attention / 100) * width)
        color = "#4CAF50" if attention > 70 else "#FFC107" if attention > 40 else "#FF5252"
        self.progress_canvas.create_rectangle(0, 0, progress_width, height, fill=color, outline="")
        
        # Add text overlay on progress bar for better readability
        self.progress_canvas.create_text(width//2, height//2, text=f"{attention:.1f}%", 
                                        fill="#FFFFFF", font=("Helvetica", 10, "bold"))

    # Update drowsiness alert display
    def update_drowsiness(self, alert):
        if alert == "DROWSY":
            self.drowsiness_label.config(text="⚠️ DROWSINESS DETECTED", fg="#FF5252", font=("Helvetica", 14, "bold"))
            self.status_label.config(text="⚠️ CRITICAL: Driver showing signs of drowsiness", fg="#FF5252")
        elif alert == "WARNING":
            self.drowsiness_label.config(text="👁️ EYES CLOSING", fg="#FFA000", font=("Helvetica", 14, "bold"))
            self.status_label.config(text="⚠️ Warning: Driver's eyes are closing", fg="#FFA000")
        elif alert == "SLEEPY":
            self.drowsiness_label.config(text="😴 SLEEPINESS DETECTED", fg="#FFC107", font=("Helvetica", 14, "bold"))
            self.status_label.config(text="Driver is yawning", fg="#FFC107")
        elif alert == "LOW_ATTENTION":
            self.drowsiness_label.config(text="📱 LOW ATTENTION", fg="#FFA000", font=("Helvetica", 14, "bold"))
            self.status_label.config(text="Driver may be on call", fg="#FFA000")
        else:
            self.drowsiness_label.config(text="✅ Normal", fg="#4CAF50", font=("Helvetica", 14, "bold"))
            self.status_label.config(text="Driver is alert and attentive", fg="#4CAF50")
    
    # Update transcription display
    def update_transcription(self, transcription):
        if transcription:
            # Insert new text with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
            self.transcription_text.insert(tk.END, timestamp + transcription + "\n")
            self.transcription_text.see(tk.END)
            # Limit the number of lines to prevent memory issues
            lines = self.transcription_text.get("1.0", tk.END).splitlines()
            if len(lines) > 30:  # Keep only the last 30 lines to minimize space
                self.transcription_text.delete("1.0", "2.0")
            # Highlight the new line
            self.transcription_text.tag_configure("recent", background="#333333")
            self.transcription_text.tag_add("recent", "end-2l", "end-1l")
            
            # Auto-adjust height based on content (between 8-12 lines)
            line_count = min(max(len(lines), 8), 12)
            self.transcription_text.config(height=line_count)

    # Update video display with responsive sizing
    def update_video(self, frame):
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Get the video label dimensions
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        # Only resize if label has been rendered and has valid dimensions
        if label_width > 1 and label_height > 1:
            # Calculate aspect ratio of the image
            img_width, img_height = pil_image.size
            aspect_ratio = img_width / img_height
            
            # Calculate new dimensions to fit within label while maintaining aspect ratio
            # But ensure we don't exceed half the window width
            window_width = self.master.winfo_width()
            max_width = min(label_width, window_width // 2 - 30)  # Half window width minus padding
            max_height = label_height
            
            if max_width / max_height > aspect_ratio:
                new_height = min(max_height, label_height)
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = min(max_width, label_width)
                new_height = int(new_width / aspect_ratio)
            
            # Ensure minimum size but allow it to scale down
            new_width = max(100, min(new_width, max_width))
            new_height = max(75, min(new_height, max_height))
            
            # Resize the image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # Update the label
        self.video_label.configure(image=self.photo_image)
        self.video_label.configure(anchor="center")

    # Show offensive language warning
    def show_offensive_warning(self, message):
        # Force the label to be visible with high-contrast colors
        self.offensive_warning_label.config(
            text="🚨 " + message,
            font=("Helvetica", 20, "bold"),
            fg="#FFFFFF",
            bg="#F44336",  # Bright red
            relief=tk.RAISED,
            bd=5
        )
        # Add visual emphasis
        self.offensive_warning_label.grid(row=0, column=0, sticky="ew", pady=5)
        # Flash effect to grab attention
        self._flash_alert(self.offensive_warning_label, "#F44336", "#FF5252", 8)
        # Auto-clear after 10 seconds
        self.master.after(10000, self._clear_offensive_warning)
    
    # Show help request warning
    def show_help_warning(self, message):
        # Force the label to be visible with high-contrast colors
        self.help_warning_label.config(
            text="🆘 " + message,
            font=("Helvetica", 20, "bold"),
            fg="#FFFFFF",
            bg="#FF5722",
            relief=tk.RAISED,
            bd=5
        )
        # Add visual emphasis
        self.help_warning_label.grid(row=1, column=0, sticky="ew", pady=5)
        # Flash effect to grab attention
        self._flash_alert(self.help_warning_label, "#FF5722", "#FF9800", 8)
        # Auto-clear after 10 seconds
        self.master.after(10000, self._clear_help_warning)
        
    # Flash alert effect
    def _flash_alert(self, label, color1, color2, count):
        if count > 0:
            current_color = label.cget("bg")
            new_color = color1 if current_color == color2 else color2
            label.config(bg=new_color)
            self.master.after(300, self._flash_alert, label, color1, color2, count-1)
    
    # Clear offensive warning - MAKE IT VISIBLE AGAIN
    def _clear_offensive_warning(self):
        self.offensive_warning_label.config(
            text="Monitoring for offensive language", 
            font=("Helvetica", 16, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E",
            relief=tk.RAISED, 
            bd=3
        )
        self.offensive_warning_label.grid(row=0, column=0, sticky="ew", pady=5)
    
    # Clear help warning - MAKE IT VISIBLE AGAIN
    def _clear_help_warning(self):
        self.help_warning_label.config(
            text="Monitoring for help requests", 
            font=("Helvetica", 16, "bold"), 
            fg="#4FC3F7", 
            bg="#1E1E1E",
            relief=tk.RAISED, 
            bd=3
        )
        self.help_warning_label.grid(row=1, column=0, sticky="ew", pady=5)
    

    def on_resize(self, event):
        # Update progress bar when window is resized
        if hasattr(self, 'last_attention') and event.widget == self.master:
            self.master.after_idle(self._update_attention_progress, self.last_attention)

# Main entry point
def main():
    root = tk.Tk()
    gui = HybridAttentionSpeechGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()