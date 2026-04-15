import logging
import sys

from PyQt6.QtWidgets import QApplication

from config import device
from ui.main_window import MainWindow

# Set up a simple console output so we can see what's happening under the hood
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    """The main entry point that wakes up the entire application."""
    
    # Create the 'App' object which handles the behind-the-scenes window logic
    app = QApplication(sys.argv)
    
    # Spin up our main interface, passing in the hardware device (CPU or GPU)
    window = MainWindow(device)
    
    # Make the window visible on the user's screen
    window.show()
    
    # Start the application's event loop (waiting for clicks, drags, etc.)
    # The script will stay alive here until the user closes the window.
    sys.exit(app.exec())


if __name__ == "__main__":
    # Standard Python check to ensure this script runs only if it's the one being executed
    main()