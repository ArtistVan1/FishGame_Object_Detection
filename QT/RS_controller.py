from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QPlainTextEdit
from PyQt5 import uic
import sys

def run():
	print("run program")
def stop():
	sys.exit()

app = QApplication([])
ui = QUiLoader().load('RS_controller.ui')
ui.button_run.clicked.connect(run)
ui.button_stop.clicked.connect(stop)

ui.show()
app.exec_()
