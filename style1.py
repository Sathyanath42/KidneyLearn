from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow, QInputDialog, QLineEdit
import sys
import time
import DecisionTreefinal
import Import_file as file
import svmkidneyrev2
import KNN_Final_CSV


class Ui_MainWindow(object):

    def setupUi(self, QMainWindow):
        QMainWindow.setObjectName("MainWindow")
        QMainWindow.resize(800, 705)
        self.centralwidget = QtWidgets.QWidget(QMainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setGeometry(QtCore.QRect(300, 100, 211, 81))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        self.pushButton1.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        self.pushButton1.setFont(font)
        self.pushButton1.setObjectName("pushButton1")

        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(60, 220, 181, 81))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setUnderline(True)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")

        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setGeometry(QtCore.QRect(12, 110, 300, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(False)
        font.setUnderline(True)
        self.label4.setFont(font)
        self.label4.setObjectName("label4")

        self.pushButton_Gini = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Gini.setGeometry(QtCore.QRect(70, 340, 141, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_Gini.setFont(font)
        self.pushButton_Gini.setObjectName("pushButton_Gini")

        self.pushButton_En = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_En.setGeometry(QtCore.QRect(70, 460, 141, 81))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_En.setFont(font)
        self.pushButton_En.setObjectName("pushButton_En")

        self.pushButton_k5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_k5.setGeometry(QtCore.QRect(350, 340, 101, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_k5.setFont(font)
        self.pushButton_k5.setObjectName("pushButton_k5")

        self.pushButton_k7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_k7.setGeometry(QtCore.QRect(350, 450, 101, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_k7.setFont(font)
        self.pushButton_k7.setObjectName("pushButton_k7")

        self.pushButton_k9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_k9.setGeometry(QtCore.QRect(350, 560, 101, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_k9.setFont(font)
        self.pushButton_k9.setObjectName("pushButton_k9")

        self.pushButton_svm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_svm.setGeometry(QtCore.QRect(610, 340, 131, 101))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_svm.setFont(font)
        self.pushButton_svm.setObjectName("pushButton_svm")

        self.pushButton_svm2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_svm2.setGeometry(QtCore.QRect(610, 480, 131, 101))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.pushButton_svm2.setFont(font)
        self.pushButton_svm2.setObjectName("pushButton_svm2")

        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(370, 230, 81, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setUnderline(True)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")

        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(640, 220, 131, 81))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setUnderline(True)
        self.label3.setFont(font)
        self.label3.setObjectName("label3")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(290, 10, 331, 61))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        self.label.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(22)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")



        MainWindow.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        # self.menubar.setObjectName("menubar")
        # self.menuFile = QtWidgets.QMenu(self.menubar)
        # self.menuFile.setObjectName("menuFile")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)
        # self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.pushButton1.setText(_translate("MainWindow", "Import File"))
        # self.pushButton1.clicked.connect(self.showDialog)
        self.pushButton1.clicked.connect(file.import_files)
        self.pushButton1.clicked.connect(self.msgbox1)

        self.label1.setText(_translate("MainWindow", "Decision Tree"))

        self.label4.setText(_translate("MainWindow", "Import file before running the algorithms!"))

        self.pushButton_Gini.setText(_translate("MainWindow", "GINI"))
        self.pushButton_Gini.clicked.connect(DecisionTreefinal.print_Gini)

        self.pushButton_En.setText(_translate("MainWindow", "ENTROPY"))
        self.pushButton_En.clicked.connect(DecisionTreefinal.print_Entropy)

        self.pushButton_k5.setText(_translate("MainWindow", "K = 3"))
        self.pushButton_k5.clicked.connect(KNN_Final_CSV.KNearestClassifer3)

        self.pushButton_k7.setText(_translate("MainWindow", "K = 5"))
        self.pushButton_k7.clicked.connect(KNN_Final_CSV.KNearestClassifer5)

        self.pushButton_k9.setText(_translate("MainWindow", "K = 7"))
        self.pushButton_k9.clicked.connect(KNN_Final_CSV.KNearestClassifer7)

        self.pushButton_svm.setText(_translate("MainWindow", "Linear"))
        self.pushButton_svm.clicked.connect(svmkidneyrev2.print_svm)

        self.pushButton_svm2.setText(_translate("MainWindow", "RBF"))
        self.pushButton_svm2.clicked.connect(svmkidneyrev2.print_rbf)

        self.label2.setText(_translate("MainWindow", "KNN"))

        self.label3.setText(_translate("MainWindow", "SVM"))

        self.label.setText(_translate("MainWindow", "Machine Learning"))

        # self.menuFile.setTitle(_translate("MainWindow", "File"))

    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter your name:')


    def msgbox1(self):
        msgbx = QMessageBox()
        msgbx.setWindowTitle("Message")
        msgbx.setText("Successful Import")
        msgbx.setIcon(QMessageBox.Information)
        msgbx.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgbx.setDefaultButton(QMessageBox.Ok)
        msgbx.setInformativeText(" ")
        x = msgbx.exec()

    def update(self):
        self.label1.adjustSize()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
