''' __main__.py '''

import sys

from PySide6.QtWidgets import QApplication

from . import gui

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = gui.MainWindow()
    widget.show()
    sys.exit(app.exec())
