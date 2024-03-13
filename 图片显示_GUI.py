import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MainWindow(QMainWindow):
    def __init__(self, ):
        super(QMainWindow, self).__init__()
        self.number = 0

        w = QWidget()
        self.setCentralWidget(w)

        self.topFiller = QWidget()
        self.topFiller.setMinimumSize(250, 2000)  #######设置滚动条的尺寸

        lab1 = QLabel(self.topFiller)
        lab1.setPixmap(QPixmap('show.png'))

        # lab2 = QLabel(self.topFiller)
        # lab2.setPixmap(QPixmap('demo.jpg'))
        # lab2.move(0, 220)

        ##创建一个滚动条
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.topFiller)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.scroll)
        w.setLayout(self.vbox)

        self.statusBar().showMessage("底部信息栏")
        self.resize(600, 500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
