#尝试加入截取目标的代码
import os
import sys
from pathlib import Path
import cv2
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.augmentations import letterbox
from utils.plots import plot_one_box, save_one_box
from models.common import DetectMultiBackend
#下面是图片显示引用的库
from PyQt5.QtWidgets import QListWidgetItem, QListView
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize
FrameIdxRole = Qt.UserRole + 1


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = False

        name = 'exp'
        save_file = ROOT / 'result'
        self.save_file = increment_path(Path(save_file) / name, exist_ok=False, mkdir=True)

        cudnn.benchmark = True
        weights = 'weights/best.pt'   # 模型加载路径
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.25  # NMS置信度
        self.iou_thres = 0.45  # IOU阈值

        # 载入模型
        self.model = DetectMultiBackend(weights, device=self.device)
        stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()  # to FP16

        # 从模型中获取各类别名称
        self.names = self.model.names
        # 给每一个类别初始化颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1122, 749)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 1061, 691))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_mainwindow = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_mainwindow.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_mainwindow.setObjectName("gridLayout_mainwindow")
        self.frame_1 = QtWidgets.QFrame(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(6)
        sizePolicy.setVerticalStretch(7)
        sizePolicy.setHeightForWidth(self.frame_1.sizePolicy().hasHeightForWidth())
        self.frame_1.setSizePolicy(sizePolicy)
        self.frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setObjectName("frame_1")
        self.label_1 = QtWidgets.QLabel(self.frame_1)
        self.label_1.setGeometry(QtCore.QRect(10, 10, 681, 461))
        self.label_1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_1.setTextFormat(QtCore.Qt.AutoText)
        self.label_1.setObjectName("label_1")
        self.gridLayout_mainwindow.addWidget(self.frame_1, 0, 0, 1, 1)

        # 创建frame_2并设置其基本属性（图片显示功能GUI）
        self.frame_2 = QtWidgets.QFrame(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        # 创建slider, listWidget, pushButton
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.listWidgetImages = QtWidgets.QListWidget()
        self.listWidgetImages.setDragEnabled(True)
        self.listWidgetImages.setMovement(QtWidgets.QListView.Static)
        self.listWidgetImages.setFlow(QtWidgets.QListView.LeftToRight)
        self.listWidgetImages.setResizeMode(QtWidgets.QListView.Adjust)
        self.listWidgetImages.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidgetImages.setModelColumn(0)
        self.listWidgetImages.setObjectName("listWidgetImages")
        self.pushButton_1 = QtWidgets.QPushButton("Button")
        # 创建垂直布局
        verticalLayout = QtWidgets.QVBoxLayout()
        # 将组件添加到垂直布局中
        verticalLayout.addWidget(self.slider)
        verticalLayout.addWidget(self.listWidgetImages)
        verticalLayout.addWidget(self.pushButton_1)
        # 将垂直布局设置到frame_2中
        self.frame_2.setLayout(verticalLayout)
        # 将frame_2添加到gridLayout_mainwindow中
        self.gridLayout_mainwindow.addWidget(self.frame_2, 0, 1, 1, 1)

        self.frame_3 = QtWidgets.QFrame(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setGeometry(QtCore.QRect(10, 180, 271, 21))
        self.label_3.setObjectName("label_3")
        self.layoutWidget1 = QtWidgets.QWidget(self.frame_3)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 10, 311, 101))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_button = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_button.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_button.setObjectName("gridLayout_button")
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_button.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_button.addWidget(self.pushButton_2, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_button.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_button.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.gridLayout_mainwindow.addWidget(self.frame_3, 1, 0, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.layoutWidget)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 331, 181))
        self.label_4.setStyleSheet("background-color: rgb(204, 204, 204);")
        self.label_4.setObjectName("label_4")
        self.gridLayout_mainwindow.addWidget(self.frame_4, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "dome_test0226"))
        self.label_1.setText(_translate("MainWindow", "TextLabel"))
        #self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow",
                                        "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">测控20-3 陈锐涛 20034500328</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "开启YOLOv5"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭YOLOv5"))
        self.pushButton_3.setText(_translate("MainWindow", "截取目标"))
        self.pushButton_4.setText(_translate("MainWindow", "OpenCV检测"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton.clicked.connect(self.button_camera_open)
        self.pushButton_2.clicked.connect(self.button_camera_close)
        self.timer_video.timeout.connect(self.show_video_frame)
        #下面这是显示图片的按钮pushButton_1信号与槽
        self.pushButton_1.clicked.connect(self.display)


    def init_logo(self):
        pix = QtGui.QPixmap('')   # 绘制初始化图片
        self.label_1.setScaledContents(True)
        self.label_1.setPixmap(pix)

    # 退出提示
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter(str(Path(self.save_file / 'camera_prediction.avi')), cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)

    def button_camera_close(self):
        self.timer_video.stop()
        self.cap.release()
        self.out.release()
        self.label_1.clear()
        self.init_logo()
        # self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):
        name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)  # 打印各目标+置信度
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

                        #这是图片目标截取的代码
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('s'):  # 如果按下's'键
                            print("开始截取")
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)
                                #save_one_box(xyxy, showimg, file=ROOT / 'result' / 'crops' / name_list[c] / f'{p.stem}.jpg', BGR=True)
                                save_one_box(xyxy, img, file=self.save_file, BGR=True)
                            print("截取成功")

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_1.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label_1.clear()
            #self.pushButton_camera.setDisabled(False)
            self.init_logo()


    #下面是图片显示功能的函数
    def display(self):
        self.listWidgetImages.setViewMode(QListView.IconMode)
        self.listWidgetImages.setModelColumn(1)  # 如果使用模型，确保列设置正确
        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)

        # slider配置
        self.slider.valueChanged.connect(self.onSliderPosChanged)

        # 替换为你想要显示图片的文件夹路径
        image_folder_path = 'E:/yolov5-70--py-qt5-master/blot'

        # 遍历文件夹中的所有文件
        for filename in os.listdir(image_folder_path):
            # 检查文件是否是图片（这里仅作为示例，可以根据需要添加更多条件）
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(image_folder_path, filename)
                image = cv2.imread(image_path)

                if image is not None:
                    self.add_image_thumbnail(image, filename, "")

    def add_image_thumbnail(self, image, frameIdx, name):
        self.listWidgetImages.itemSelectionChanged.disconnect(self.onItemSelectionChanged)

        height, width, channels = image.shape
        bytes_per_line = width * channels
        qImage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImage)

        item = QListWidgetItem(QIcon(pixmap), str(frameIdx) + ": " + name)
        item.setData(FrameIdxRole, frameIdx)

        self.listWidgetImages.addItem(item)

        self.listWidgetImages.setCurrentRow(self.listWidgetImages.count() - 1)

        self.listWidgetImages.itemSelectionChanged.connect(self.onItemSelectionChanged)

    def onItemSelectionChanged(self):
        pass

    def onSliderPosChanged(self, value):
        self.listWidgetImages.setIconSize(QSize(value, value))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口透明度
    ui.setWindowOpacity(0.93)
    # 去除顶部边框
    # ui.setWindowFlags(Qt.FramelessWindowHint)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    # 设置应用图标
    ui.setWindowIcon(icon)
    ui.show()
    sys.exit(app.exec_())
