
import sys
import cv2


from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtWidgets import QWidget, QListWidgetItem, QListView
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSignal


from ui_ImageBrowserWidget1 import Ui_Form

"显示多张图片的缩略图 加滚动条"

FrameIdxRole = Qt.UserRole + 1


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)

        self.setupUi(self)

        self.pushButton.clicked.connect(self.display)


    def display(self):
        self.listWidgetImages.setViewMode(QListView.IconMode)
        self.listWidgetImages.setModelColumn(1)
        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)

        # slider
        self.sliderScale.valueChanged.connect(self.onSliderPosChanged)

        for i in range(100):
            image = cv2.imread('E:/yolov5-70--py-qt5-master/blot/02.jpg')
            self.add_image_thumbnail(image,"图片",str(i))


    def add_image_thumbnail(self, image, frameIdx, name):
        self.listWidgetImages.itemSelectionChanged.disconnect(self.onItemSelectionChanged)

        height, width, channels = image.shape
        print(image.shape)
        bytes_per_line = width * channels
        print(bytes_per_line)
        qImage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)

        item = QListWidgetItem(QIcon(pixmap), str(frameIdx) + ": " + name)
        item.setData(FrameIdxRole, frameIdx)

        self.listWidgetImages.addItem(item)

        # to bottom
        # self.listWidgetImages.scrollToBottom()
        self.listWidgetImages.setCurrentRow(self.listWidgetImages.count() - 1)

        print('\033[32;0m  --- add image thumbnail: {}, {} -------'.format(frameIdx, name))

        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)
        # self.listWidgetImages.it

    def resizeEvent(self, event):
        width = self.listWidgetImages.contentsRect().width()
        self.sliderScale.setMaximum(width)
        self.sliderScale.setValue(width - 40)

    def onItemSelectionChanged(self):
        pass

    def onSliderPosChanged(self, value):
        self.listWidgetImages.setIconSize(QSize(value, value))


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
