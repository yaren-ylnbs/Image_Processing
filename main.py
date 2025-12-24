# ======================== main.py ========================
"""
Application entry point
"""
import tkinter as tk
from main_app import ImageProcessingApplication

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ImageProcessingApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()
