#这是一些小的demo测试的代码存放区
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建一个 QVBoxLayout
        layout = QVBoxLayout()

        # 创建一个 QTextEdit 控件用于显示文本
        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(True)  # 设置为只读，如果你不希望用户编辑文本
        layout.addWidget(self.textEdit)

        # 创建一个 QPushButton 控件
        self.button = QPushButton('按下我', self)
        self.button.clicked.connect(self.on_button_clicked)  # 连接按钮的点击信号到槽函数
        layout.addWidget(self.button)

        # 设置窗口的主布局
        self.setLayout(layout)

        # 设置窗口的标题和尺寸
        self.setWindowTitle('按钮点击示例')
        self.setGeometry(300, 300, 300, 200)

    def on_button_clicked(self):
        # 在 QTextEdit 控件中添加文本
        self.textEdit.append('按钮被按下')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())