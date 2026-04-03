# main.py
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from app.gui.main_window import Application

if __name__ == "__main__":
    app = Application()
    app.mainloop()
