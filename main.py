"""
Run this code Only for the presentation.
"""

# Taha Faghani (99542321)
# AI Final Project 
# Dr.Salehi

# Please make sure you install Qt (with this command):
# pip install PyQt5

"""
This is the main code of the project where the models are connected here.
Please do not change anything from this code or the folders.
"""

import os
import io
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QDialog, QTextEdit, QDesktopWidget, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

# Import machine learning model scripts from the 'models' folder
from models import ann
from models import svm
from models import dt
from models import knn
from models import kmeans
from models import logistic_regression as lr
from models import naive_base as nb
from models import best  # Assuming 'best' is a model script to determine the best method

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Define a custom QPushButton class with an icon and text
class IconButton(QWidget):
    clicked = pyqtSignal()  # Define a custom signal for the button click

    def __init__(self, icon_path, text, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 150)  # Set a fixed size for the button

        layout = QVBoxLayout()
        self.icon_label = QLabel(self)
        
        # Load the icon image from the specified path
        pixmap = QPixmap(icon_path)
        if pixmap.isNull():
            print(f"Failed to load icon: {icon_path}")
        else:
            pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel(text, self)
        self.text_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)  # Set spacing between icon and text
        self.setLayout(layout)

    # Override mousePressEvent to emit the custom signal when clicked
    def mousePressEvent(self, event):
        self.clicked.emit()

# Define a custom QDialog to display the output of the models
class OutputWindow(QDialog):
    def __init__(self, title, output, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)  # Set the window title
        self.resize(400, 300)  # Resize the window
        self.init_ui(output)  # Initialize the UI components
        
        # Set the background image
        icons_path = resource_path('icons')
        self.set_background_image(resource_path("icons/background1.png"))
        self.move_to_left()  # Move the window to the left side of the screen

    def init_ui(self, output):
        layout = QVBoxLayout()
        self.background_label = QLabel(self)
        layout.addWidget(self.background_label)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # Make the text edit read-only
        self.text_edit.setText(output)  # Set the output text
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    # Set the background image of the window
    def set_background_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Failed to load background image: {image_path}")
        else:
            self.background_label.setPixmap(pixmap)
            self.background_label.setScaledContents(True)
            self.background_label.lower()  # Ensure the background is behind other widgets

    # Move the window to the left side of the screen
    def move_to_left(self):
        screen = QDesktopWidget().screenGeometry()
        x = 0
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

# Define the main window with model buttons
class MyWindow(QWidget):
    def __init__(self, buttons, parent=None):
        super().__init__(parent)
        self.buttons = buttons  # List of button numbers to be displayed
        self.current_output_window = None  # Track the currently displayed output window
        self.init_ui()

    def init_ui(self):
        self.resize(80, 170)  # Resize the window 80,200
        main_layout = QVBoxLayout(self)

        self.background_label = QLabel(self)
        icons_path = resource_path('icons')
        self.set_background_image(resource_path("icons/bac_s.png"))  # Set the background image
        self.background_label.setScaledContents(True)

        content_widget = QWidget(self)
        layout = QGridLayout(content_widget)
        layout.setAlignment(Qt.AlignCenter)  # Center the buttons in the layout

        # Add buttons based on the provided list of button numbers
        if 1 in self.buttons:
            self.button1 = IconButton(resource_path('icons/icon1.png'), '  پوریا  \nSVM', self)
            self.button1.clicked.connect(self.run_svm)
            layout.addWidget(self.button1, 0, 0)

        if 2 in self.buttons:
            self.button2 = IconButton(resource_path('icons/icon2.png'), 'پوریا  \n ANN', self)
            self.button2.clicked.connect(self.run_ann)
            layout.addWidget(self.button2, 0, 1)

        if 3 in self.buttons:
            self.button3 = IconButton(resource_path('icons/icon3.png'), 'پوریا \n Decision Tree', self)
            self.button3.clicked.connect(self.run_dt)
            layout.addWidget(self.button3, 0, 2)

        if 4 in self.buttons:
            self.button4 = IconButton(resource_path('icons/icon4.png'), 'پوریا \n Logistic regression', self)
            self.button4.clicked.connect(self.run_lr)
            layout.addWidget(self.button4, 1, 0)

        if 5 in self.buttons:
            self.button5 = IconButton(resource_path('icons/icon5.png'), 'پوریا \n Naive Base', self)
            self.button5.clicked.connect(self.run_nb)
            layout.addWidget(self.button5, 1, 1)

        if 6 in self.buttons:
            self.button6 = IconButton(resource_path('icons/icon6.png'), 'پوریا \n KNN', self)
            self.button6.clicked.connect(self.run_knn)
            layout.addWidget(self.button6, 1, 2)

        if 7 in self.buttons:
            self.button7 = IconButton(resource_path('icons/icon7.png'), 'پوریا \n KMEANS', self)
            self.button7.clicked.connect(self.run_kmeans)
            layout.addWidget(self.button7, 2, 1)

        if 8 in self.buttons:
            self.button8 = IconButton(resource_path('icons/icon8.png'), 'بهترین انتخاب پوریا', self)
            self.button8.clicked.connect(self.run_best)
            layout.addWidget(self.button8, 2, 1)

        main_layout.addWidget(self.background_label)
        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)

    # Set the background image of the window
    def set_background_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Failed to load background image: {image_path}")
        else:
            self.background_label.setPixmap(pixmap)
            self.background_label.setScaledContents(True)
            self.background_label.lower()

    # Define the methods to run the corresponding model scripts
    def run_svm(self):
        print("The SVM Model is running... Please wait!")
        self.run_model(svm, "SVM results")

    def run_ann(self):
        print("The ANN model is running... Please wait!")
        self.run_model(ann, "ANN Results")

    def run_dt(self):
        print("The Decision Tree model is running... Please wait!")
        self.run_model(dt, "Decision Tree Results")

    def run_lr(self):
        print("The Logistic Regression model is running... Please wait!")
        self.run_model(lr, "Logistic regression Results")

    def run_nb(self):
        print("The Naive Base model is running... Please wait!")
        self.run_model(nb, "Naive base Results")

    def run_knn(self):
        print("The KNN model is running... Please wait!")
        self.run_model(knn, "KNN Results")

    def run_kmeans(self):
        print("The KMEANS model is running... Please wait!")
        self.run_model(kmeans, "Kmeans Situation")

    def run_best(self):
        print("The Best Model is Evaluating... Please wait!")
        self.run_model(best, "Best Method For Dataset")

    # Run the specified script and display its output
    def run_model(self, script_module, title):
        if self.current_output_window:
            self.current_output_window.close()

        # Redirect stdout to capture script output
        sys.stdout = io.StringIO()
        try:
            script_module.runscript()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        self.show_output_window(title, output)

    # Show the output in a new window
    def show_output_window(self, title, output):
        self.current_output_window = OutputWindow(title, output)
        self.current_output_window.show()

# Define the main application window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.current_output_window = None  # Keep track of current output window
        self.init_ui()

    def init_ui(self):
        self.resize(800, 400)  # Resize the window
        self.setWindowTitle("                                            Welcome !!!  Please Choose Dataset and the Method")

        main_layout = QVBoxLayout(self)

        self.background_label = QLabel(self)
        icons_path = resource_path('icons')
        self.set_background_image(resource_path("icons/background111.png"))
        self.background_label.setScaledContents(True)

        content_widget = QWidget(self)
        layout = QGridLayout(content_widget)
        layout.setAlignment(Qt.AlignCenter)  # Center the buttons in the layout

        # Add the supervised button
        self.supervised_button = IconButton(resource_path('icons/icon_supervised.png'), 'با نظارت پوریا \n Supervised', self)
        self.supervised_button.clicked.connect(self.show_supervised_window)
        layout.addWidget(self.supervised_button, 0, 0)

        # Add the unsupervised button
        self.unsupervised_button = IconButton(resource_path('icons/icon_unsupervised.png'), 'بدون نظارت پوریا\n unsupervised', self)
        self.unsupervised_button.clicked.connect(self.show_unsupervised_window)
        layout.addWidget(self.unsupervised_button, 0, 1)

        # Add the file upload button
        self.file_upload_button = IconButton(resource_path('icons/icon_upload.png'), 'Upload CSV File', self)
        self.file_upload_button.clicked.connect(self.upload_csv)
        layout.addWidget(self.file_upload_button, 1, 0)

        # Add the checkbox for default dataset
        self.default_dataset_checkbox = QCheckBox('Continue with default dataset(churn.csv)', self)
        self.default_dataset_checkbox.stateChanged.connect(self.toggle_upload_button)
        layout.addWidget(self.default_dataset_checkbox, 1, 1)

        main_layout.addWidget(self.background_label)
        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)

        self.update_button_state()

    # Set the background image of the main window
    def set_background_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Failed to load background image: {image_path}")
        else:
            self.background_label.setPixmap(pixmap)
            self.background_label.setScaledContents(True)
            self.background_label.lower()

    # Show the supervised window with model buttons
    def show_supervised_window(self):
        print("Supervised button clicked")
        self.supervised_window = MyWindow(buttons=[1, 2, 3, 4, 5, 6, 8])
        self.supervised_window.show()
        self.move_right(self.supervised_window)

    # Show the unsupervised window with model buttons
    def show_unsupervised_window(self):
        print("Unsupervised button clicked")
        self.unsupervised_window = MyWindow(buttons=[7])
        self.unsupervised_window.show()
        self.move_right(self.unsupervised_window)

    # Upload a CSV file
    def upload_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            print(f"CSV file selected: {file_path}")
            self.file_upload_button.setProperty('selected', True)
            self.update_button_state()

    # Toggle the upload button based on the checkbox state
    def toggle_upload_button(self, state):
        if state == Qt.Checked:
            self.file_upload_button.setEnabled(False)
        else:
            self.file_upload_button.setEnabled(True)
        self.update_button_state()

    # Update the state of the supervised and unsupervised buttons
    def update_button_state(self):
        if self.file_upload_button.property('selected') or self.default_dataset_checkbox.isChecked():
            self.supervised_button.setEnabled(True)
            self.unsupervised_button.setEnabled(True)
            self.supervised_button.setToolTip("")
            self.unsupervised_button.setToolTip("")
        else:
            self.supervised_button.setEnabled(False)
            self.unsupervised_button.setEnabled(False)
            self.supervised_button.setToolTip("Please choose a CSV file or continue with the default dataset.")
            self.unsupervised_button.setToolTip("Please choose a CSV file or continue with the default dataset.")

    # Move the new window to the right of the main window
    def move_right(self, window):
        if self.current_output_window:
            self.current_output_window.close()
        self.current_output_window = window
        screen = QDesktopWidget().screenGeometry()
        x = self.geometry().right() - 30  # Adjust the value to move the window slightly to the right
        y = (screen.height() - window.height()) // 2
        window.move(x, y)

# Define the welcome window
class WelcomeWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.resize(400, 300)
        self.init_ui()
        
        # Set a timer to close the welcome window after 3 seconds
        QTimer.singleShot(4000, self.start_main_window)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Add a QLabel to display the welcome message
        label = QLabel("Welcome to the AI Final Project\nBy Taha Faghani", self)
        label.setAlignment(Qt.AlignCenter)
        
        # Add a QLabel to display the animated GIF
        self.gif_label = QLabel(self)
        gif_path = resource_path('icons/welcome.gif')
        movie = QMovie(gif_path)
        self.gif_label.setMovie(movie)
        movie.start()
        
        layout.addWidget(label)
        layout.addWidget(self.gif_label)
        self.setLayout(layout)

    def start_main_window(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

if __name__ == '__main__':
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Show the welcome window
    welcome_window = WelcomeWindow()
    welcome_window.show()

    # Start the event loop
    sys.exit(app.exec_())
