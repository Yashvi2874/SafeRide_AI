import tkinter as tk
from tkinter import scrolledtext

# Main GUI class
class HybridAttentionSpeechGUI:
    def __init__(self, master):
        self.master = master
        master.title("Hybrid Attention and Speech Project")
        master.geometry("600x550")
        master.resizable(True, True)

        # Configure styles
        self.master.option_add("*Font", "Helvetica 10")
        
        # Attention label
        self.attention_label = tk.Label(master, text="Attention Level: 0%", font=("Helvetica", 14, "bold"))
        self.attention_label.pack(pady=10)

        # Drowsiness alert
        self.drowsiness_label = tk.Label(master, text="Status: Normal", font=("Helvetica", 12))
        self.drowsiness_label.pack(pady=5)
        
        # Additional status alerts
        self.status_label = tk.Label(master, text="", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

        # Speech transcription section
        self.transcription_label = tk.Label(master, text="Speech Transcription:", font=("Helvetica", 12, "bold"))
        self.transcription_label.pack(pady=(15, 5))

        # Scrollable text area for transcription
        self.transcription_frame = tk.Frame(master)
        self.transcription_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        self.transcription_text = scrolledtext.ScrolledText(
            self.transcription_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=12, 
            font=("Helvetica", 10)
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True)

        # Offensive warning label
        self.offensive_warning_label = tk.Label(master, text="", font=("Helvetica", 12, "bold"), fg="red")
        self.offensive_warning_label.pack(pady=10)

    # Update attention display
    def update_attention(self, attention):
        self.attention_label.config(text=f"Attention Level: {attention:.1f}%")
        if attention > 70:
            self.attention_label.config(fg="green")
        elif attention > 40:
            self.attention_label.config(fg="orange")
        else:
            self.attention_label.config(fg="red")

    # Update drowsiness alert display
    def update_drowsiness(self, alert):
        if alert == "DROWSY":
            self.drowsiness_label.config(text="Status: DROWSINESS DETECTED", fg="red", font=("Helvetica", 12, "bold"))
            self.status_label.config(text="", fg="black")
        elif alert == "WARNING":
            self.drowsiness_label.config(text="Status: EYES CLOSING", fg="orange", font=("Helvetica", 12, "bold"))
            self.status_label.config(text="", fg="black")
        elif alert == "SLEEPY":
            self.drowsiness_label.config(text="Status: SLEEPINESS DETECTED", fg="gold", font=("Helvetica", 12, "bold"))
            self.status_label.config(text="Driver is yawning", fg="gold")
        elif alert == "LOW_ATTENTION":
            self.drowsiness_label.config(text="Status: LOW ATTENTION", fg="orange", font=("Helvetica", 12, "bold"))
            self.status_label.config(text="Driver may be on call", fg="orange")
        else:
            self.drowsiness_label.config(text="Status: Normal", fg="green", font=("Helvetica", 12))
            self.status_label.config(text="", fg="black")
    
    # Update transcription display
    def update_transcription(self, transcription):
        if transcription:
            # Insert new text and auto-scroll to the end
            self.transcription_text.insert(tk.END, transcription + "\n")
            self.transcription_text.see(tk.END)
            # Limit the number of lines to prevent memory issues
            lines = self.transcription_text.get("1.0", tk.END).splitlines()
            if len(lines) > 100:  # Keep only the last 100 lines
                self.transcription_text.delete("1.0", "2.0")

    # Show offensive language warning
    def show_offensive_warning(self, message):
        self.offensive_warning_label.config(text=message)
        # Auto-clear the warning after 5 seconds
        self.master.after(5000, lambda: self.offensive_warning_label.config(text=""))

# Main entry point
def main():
    root = tk.Tk()
    gui = HybridAttentionSpeechGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()