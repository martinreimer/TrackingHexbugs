import PyQt6
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QGridLayout, QWidget, QMessageBox, QLabel, QCheckBox, \
        QSizePolicy, QComboBox
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import pyqtRemoveInputHook
import pandas as pd
import pyqtgraph as pg
import imageio as io
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
from pdb import set_trace
import cv2
import matplotlib.patches as patches


DEFAULT_DIRECTORY = os.getcwd()
video_name = None
video_filename = None
video_filepath = None
IMAGE_SIZE = None #'(640, 480)

def trace():
    pyqtRemoveInputHook()
    set_trace()


#### ROI helper class
class ROI:
    def __init__(self, pos=[100, 100], shown=True):
        """
        ROI class that keeps ROI position and if it is active.
        :param pos: list, tuple or pyqtgraph Point
        :param shown: boolean
        """
        self.pos = pos
        self.shown = shown

    def serialize(self):
        """
        Serializes object to dictionary
        :return: dict
        """
        return {'pos': tuple(self.pos), 'shown': self.shown}

    def isValid(self):
        return self.pos[0] > 0 and self.pos[1] > 0 and self.pos[0] <= IMAGE_SIZE[1] and self.pos[1] <= IMAGE_SIZE[0]

## Custom ImageView
class ImageView(pg.ImageView):
    keysignal = pyqtSignal(int)
    mousesignal = pyqtSignal(int)

    def __init__(self, im, parent=None):
        """
        Custom ImageView class to handle ROIs dynamically
        :param im: The image to be shown
        :param rois: The rois for this image
        :param parent: The parent widget where the window is embedded
        """
        # Set Widget as parent to show ImageView in Widget
        super().__init__(parent=parent)

        # Set 2D image
        self.setImage(im)
        self.colors = ['#1a87f4', '#ebf441', '#9b1a9b', '#42f489', '#f44141', '#41f4e9', '#f4a941', '#41f4a9', '#a941f4', '#f4a941']
        
        self.realRois = []
        self.textItems = []  # List to keep track of text items
        self.textBackgrounds = []  # List to keep track of text backgrounds

        for i in range(10):
            t = pg.CrosshairROI([-1, -1])
            t.setPen(pg.mkPen(self.colors[i]))
            t.aspectLocked = True
            t.rotateAllowed = False
            ### Storing, not actually saving! ###
            # t.sigRegionChanged.connect(self.saveROIs)
            self.realRois.append(t)
            self.getView().addItem(self.realRois[-1])
        
        self.getView().setMenuEnabled(False)

        # Set reference to stack
        self.stack = parent

    def mousePressEvent(self, e):
        # Important, map xy coordinates to scene!
        pos = e.pos()
        xy = self.getImageItem().mapFromScene(pos.x(), pos.y())
        modifiers = QApplication.keyboardModifiers()

        #print(f"Mouse pressed at {xy} - {pos.x()}, {pos.y()}")

        # Set posterior point only when point is on the image
        if e.button() == Qt.MouseButton.LeftButton:
        #if e.button() == Qt.MouseButton.LeftButton and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.realRois[self.stack.annotating].setPos(xy)
            self.realRois[self.stack.annotating].show() 

            # Check checkbox
            self.mousesignal.emit(self.stack.annotating)

    def setROIs(self, rois):
        """Set ROIs from a list of ROI instances

        Args:
            rois (list[ROI]): The ROIs of the current frame
        """
       # Clear previous text items and backgrounds
        for text in self.textItems:
            self.getView().removeItem(text)
        self.textItems.clear()

        for bg in self.textBackgrounds:
            self.getView().removeItem(bg)
        self.textBackgrounds.clear()

        for i, r in enumerate(rois):
            self.realRois[i].setPos(r.pos)

            if r.shown and r.isValid():
                self.realRois[i].show()
                # Create text item
                text = pg.TextItem(f"HB {i+1}", color="white", fill='r') # set fill as 'r' to get red background
                text.setPos(r.pos[0], r.pos[1])
                self.getView().addItem(text)
                self.textItems.append(text)
            else:
                self.realRois[i].hide()

    def getROIs(self):
        """Saves and returns the current ROIs"""
        
        return [ROI(r.pos(), r.isVisible()) for r in self.realRois]

    def keyPressEvent(self, ev):
        """Pass keyPressEvent to parent classes
        
        Parameters
        ----------
        ev : event
            key event
        """
        self.keysignal.emit(ev.key())


##################
### STACK
##################
class Stack(QWidget):
    def __init__(self, fn, rois=None, fps=30):
        """
        Main Widget to keep track of the stack (or movie) and the ROIs.
        :param stack: ndarray
        :param rois: None or list of saved ROIs (json)
        """
        super().__init__()

        self.fn = fn
        self.fps = fps
        self.colors = ['#1a87f4', '#ebf441', '#9b1a9b', '#42f489', '#f44141', '#41f4e9', '#f4a941', '#41f4a9', '#a941f4', '#f4a941']
        self.curId = 0
        self.freeze = False

        self.im = self.load_video(self.fn, self.fps)
        self.dim = self.im.shape        
        self.rois = self.createROIs(rois)
        self.w = ImageView(self.im.transpose(0, 2, 1, 3), parent=self)


        ### Create Grid Layout and add the main image window to layout ###
        self.l = QGridLayout()
        self.l.addWidget(self.w, 0, 0, 6, 1)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.w.setSizePolicy(sizePolicy)
        self.w.show()

        self.w.sigTimeChanged.connect(self.changeZ)

        self.w.keysignal.connect(self.keyPress)
        self.w.mousesignal.connect(self.mousePress)

        ### Checkboxes for point selection ###
        self.annotate = QComboBox()
        self.annotate.addItems([
            f"Hexbug {i}" for i in range(1,11)
        ])
        self.annotate.currentIndexChanged.connect(self.changeAnnotating)
        self.annotating = 0

        self.l.addWidget(QLabel("Currently annotating:"), 1, 1)
        self.l.addWidget(self.annotate, 2, 1)

        self.autoMove = QCheckBox("Automatically change frame")
        self.autoMove.setChecked(True)
        self.l.addWidget(self.autoMove, 4, 1)

        ### Add another empty label to ensure nice GUI formatting ###
        self.ll = QLabel()
        self.ll.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding))
        self.l.addWidget(self.ll, 5, 1)

    
        self.setLayout(self.l)

        ### Update once the checkboxes and the ROIs ###
        self.w.setROIs(self.rois[0])

    def load_video(self, filename, target_fps):
        vidcap = cv2.VideoCapture(filename)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        num_frames = int(duration * target_fps)

        frames = []
        frame_idx = 0
        success, image = vidcap.read()
        while success:
            if frame_idx % int(fps / target_fps) == 0:
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            success, image = vidcap.read()
            frame_idx += 1

        vidcap.release()
        return np.array(frames)


    def changeAnnotating(self):
        self.annotating = self.annotate.currentIndex()

    def createROIs(self, rois=None):
        tmp_rois = [[ROI([-50, -50], False) for i in range(10)]
                for _ in range(self.dim[0])]

        # Loads saved ROIs
        if type(rois) == list:
            for r in rois:
                tmp_rois[r['z']][r['id']].pos = r['pos']
                tmp_rois[r['z']][r['id']].shown = True

        return tmp_rois
           

    def mousePress(self, roi_id):
        if self.autoMove.isChecked():
            self.forceStep(1)

    def forceStep(self, direction=1):
        self.w.setCurrentIndex(self.curId+direction)

    def changeZ(self, *args, force_step=0):
        # Save ROIs
        self.rois[self.curId] = self.w.getROIs()
            

        # New image position
        self.curId = self.w.currentIndex

        # Set current image and current ROI data
        self.w.setROIs(self.rois[self.curId])


    def keyPress(self, key):
        # go next frame
        if key == Qt.Key.Key_D:
             self.forceStep(1)
        # right button click
        elif key == Qt.Key.Key_Right:
            self.forceStep(1)

        # go prev frame
        elif key == Qt.Key.Key_A:
            self.forceStep(-1)
        # left button click
        elif key == Qt.Key.Key_Left:
            self.forceStep(-1)

    def keyPressEvent(self, e):
        # Emit Save command to parent class
        if e.key() == Qt.Key.Key_S:
            self.w.keysignal.emit(e.key())




from PyQt6.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QDoubleSpinBox

class YoloExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to YOLO Format")

        self.pixel_size_label = QLabel("Pixel Size of Hexbug Head:")
        self.pixel_size_input = QDoubleSpinBox()
        self.pixel_size_input.setRange(1, 100)
        self.pixel_size_input.setValue(10)  # Default value

        self.preview_button = QPushButton("Preview (close after 3 seconds)")
        self.export_button = QPushButton("Export (wait a few secs)")

        layout = QVBoxLayout()
        layout.addWidget(self.pixel_size_label)
        layout.addWidget(self.pixel_size_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.export_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.preview_button.clicked.connect(self.preview)
        self.export_button.clicked.connect(self.export)

    def preview(self):
        pixel_size = self.pixel_size_input.value()
        self.parent().previewYoloExport(pixel_size)

    def export(self):
        pixel_size = self.pixel_size_input.value()
        self.parent().exportYoloFormat(pixel_size)
        self.close()


class OpenOpenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decide Parameters")

        # get the fps of the video
        reader = io.get_reader(video_filepath)
        meta = reader.get_meta_data()
        current_fps = meta['fps']
        video_length = meta['duration']
        global IMAGE_SIZE
        IMAGE_SIZE = (meta['size'][1], meta['size'][0])


        # let user decide the fps
        self.fps_label = QLabel(f"Current FPS is {current_fps} with video length of {video_length} s.\nChoose Frames per Second (FPS): {str(current_fps)}")
        self.fps_input = QDoubleSpinBox()
        self.fps_input.setRange(1, 400)
        self.fps_input.setValue(current_fps)  # Default value
        self.start_button = QPushButton("Start Annotation")

        layout = QVBoxLayout()
        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_input)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.start_button.clicked.connect(self.start)

    def start(self):
        fps = self.fps_input.value()
        self.parent().load_annotations(fps)
        self.close()


######################
######################
##     MAIN WINDOW  ##
######################
######################
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_fn = None
        self.status = self.statusBar()
        self.menu = self.menuBar()
        self.file = self.menu.addMenu("&File")
        self.file.addAction("Open", self.open)
        self.file.addAction("Save as .traco", self.save)
        self.file.addAction("Exit", self.close)

        self.features = self.menu.addMenu("&Features")
        self.features.addAction("Plot trajectories", self.plotTrajectories)
        self.features.addAction("Export Tracking to CSV", self.export_csv_format)
        self.features.addAction("Export Bounding Boxes YOLO Format", self.openYoloExportDialog)

        self.fn = None
        self.history = []

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("TRACO Annotator")

    def plotTrajectories(self):
        if not self.fn:
            return

        plt.figure()
        plt.imshow(self.stack.im[0])

        ts = [[] for i in range(10)]
        xys = [[] for i in range(10)]

        for i in range(self.stack.dim[0]):
            for j in range(4):
                r = self.stack.rois[i][j]

                if r.isValid():
                    ts[j].append(i)
                    xys[j].append(r.pos)

        for i, xy in enumerate(xys):
            for j in xy:
                plt.scatter(*j, color="red")

        
        plt.xlim([0, self.stack.dim[2]])
        plt.ylim([self.stack.dim[1], 0])
        plt.show()

    def export_csv_format(self):
        # specify folder where multiple files will be saved
        fn = QFileDialog.getSaveFileName(directory=DEFAULT_DIRECTORY, filter="*.csv")[0]

        if fn:
            tmp = []

            for i in range(len(self.stack.rois)):
                for j in range(10):
                    r = self.stack.rois[i][j]

                    e = {
                        't': i,
                        'hexbug': j,
                        'x': r.pos[0],
                        'y': r.pos[1]
                    }

                    if r.isValid():
                        tmp.append(e)

            pd.DataFrame(tmp).to_csv(fn)
            
            QMessageBox.information(self, "Data exported.", f"Data saved at\n{fn}")

    def openOpenDialog(self):
        dialog = OpenOpenDialog(self)
        dialog.exec() 

    def openYoloExportDialog(self):
        dialog = YoloExportDialog(self)
        dialog.exec() 


    def previewYoloExport(self, pixel_size):
        if not self.fn:
            return

        first_frame = self.stack.im[0]
        plt.imshow(first_frame)
        plt.title(f"First frame w/ bounding box pixel size: {pixel_size}")
        
        for roi in self.stack.rois[0]:
            if roi.isValid():
                plt.title(f"roi: {roi.pos}")
                # Extract the position and size of the ROI
                x, y = roi.pos
                
                # Create a rectangle patch
                rect = patches.Rectangle((x-pixel_size/2, y-pixel_size/2), pixel_size, pixel_size, linewidth=1, edgecolor='red', facecolor='none')
                
                # Add the rectangle to the plot
                plt.gca().add_patch(rect)
                
        plt.show()
        # close the plot after 3 seconds
        plt.pause(3)
        plt.close()


    def exportYoloFormat(self, pixel_size):
        yolo_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save YOLO Annotations")
        if not yolo_dir:
            return

        # Create a directory for saving frames with bounding boxes
        bounding_boxes_dir = os.path.join(yolo_dir, "bounding_boxes")
        os.makedirs(bounding_boxes_dir, exist_ok=True)
        images_dir = os.path.join(yolo_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        labels_dir = os.path.join(yolo_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)


        # Set up video writer
        video_output_path = os.path.join(yolo_dir, f"{video_name}_tracking_video.mp4")
        frame_height, frame_width = self.stack.im[0].shape[:2]
        slow_frame_rate = 10  # Set slower frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, slow_frame_rate, (frame_width, frame_height))

        for frame_idx, frame_rois in enumerate(self.stack.rois):
            yolo_data = []
            # Save img as jpg
            output_img_name = f"{video_name}_{frame_idx:03d}.jpg"
            output_img_path = os.path.join(images_dir, output_img_name)
            plt.imsave(output_img_path, self.stack.im[frame_idx])

            # Read the frame with OpenCV
            frame = cv2.imread(output_img_path)
            
            with open(os.path.join(labels_dir, f"{video_name}_{frame_idx:03d}.txt"), 'w') as f:
                for hexbug_id in range(10):
                    r = self.stack.rois[frame_idx][hexbug_id]
                    if r.isValid():
                        x_center = r.pos[0] / self.stack.dim[2]
                        y_center = r.pos[1] / self.stack.dim[1]
                        width = pixel_size / self.stack.dim[2]
                        height = pixel_size / self.stack.dim[1]
                        f.write(f"{0} {x_center} {y_center} {width} {height}\n")

                        # Convert YOLO format back to bounding box format for drawing
                        x_min = int((x_center - width / 2) * self.stack.dim[2])
                        y_min = int((y_center - height / 2) * self.stack.dim[1])
                        x_max = int((x_center + width / 2) * self.stack.dim[2])
                        y_max = int((y_center + height / 2) * self.stack.dim[1])

                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        # Draw label
                        label = f"HB {hexbug_id + 1}"
                        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x_min, y_min - label_height - 10), (x_min + label_width, y_min), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save the frame with bounding boxes
            bounding_box_img_name = f"{video_name}_bbox_{frame_idx:03d}.jpg"
            bounding_box_img_path = os.path.join(bounding_boxes_dir, bounding_box_img_name)
            cv2.imwrite(bounding_box_img_path, frame)

            # Duplicate each frame to slow down the video further
            for _ in range(3):  # Adjust this value to make it slower
                video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        QMessageBox.information(self, "Export Complete", f"YOLO annotations, video, and frames with bounding boxes saved in {yolo_dir}")


    def close(self):
        ok = QMessageBox.question(self, "Exiting?",
            "Do you really want to exit the annotation program? Ensure you save for progress.")

        if ok == QMessageBox.Yes:
            super().close()

    def connectROIs(self):
        for i in range(len(self.stack.w.realRois)):
            self.stack.w.realRois[i].sigRegionChanged.connect(self.p)

    def updateStatus(self):
        """Shows the current "z-value", i.e. the image ID, and its dimensions
        """
        self.status.showMessage('z: {} x: {} y: {}'.format(self.stack.w.currentIndex,
                                                           self.stack.dim[0],
                                                           self.stack.dim[1]))

    def open(self):
        # Select a file
        fn = QFileDialog.getOpenFileName(directory=DEFAULT_DIRECTORY)[0]
        # set video file name as global variable
        global video_filename, video_name, video_filepath
        # for filename remove the path, we only need the name
        video_filepath = fn
        video_filename = os.path.basename(fn)
        video_name = video_filename.replace(".mp4", "").replace(".MOV", "").replace(".mov", "")
        self.status.showMessage(fn)

        # Was a file selected? Go for it!
        if fn:
            self.fn = fn # assuming these are mp4 files...

            OpenOpenDialog(self).exec()

            
    def load_annotations(self, fps):
        # if self.fn ends with mp4 or mov, replace it with traco
        self.fn_rois = self.fn.replace(".mp4", ".traco").replace(".MOV", ".traco").replace(".mov", ".traco")

        # If ROI file is existing, read and decode
        if os.path.isfile(self.fn_rois):
            with open(self.fn_rois, 'r') as fp:
                rois = json.load(fp)['rois']

        else:
            rois = None

        # Create new Image pane and show first image,
        # connect slider and save function
        self.stack = Stack(self.fn, rois=rois, fps=int(fps))
        self.setCentralWidget(self.stack)

        self.stack.w.sigTimeChanged.connect(self.updateStatus)
        self.stack.w.keysignal.connect(self.savekeyboard)

        self.connectROIs()

        self.setWindowTitle("TRACO Annotator | Working on file {}".format(self.fn))


    def p(self, e):
        """Shows current position
        
        Parameters
        ----------
        e : event
            Mouse event carrying the position
        """
        self.status.showMessage("{}".format(e.pos()))

    def save(self):
        """Saves all ROIs to file
        """
        if self.fn_rois:
            with open(self.fn_rois, "w") as fp:
                json.dump({
                    "rois": [{'z': i,
                              'id': j,
                              'pos': self.stack.rois[i][j].serialize()['pos']}
                             for i in range(len(self.stack.rois))
                             for j in range(len(self.stack.rois[i]))
                             if self.stack.rois[i][j].isValid()]
                }, fp, indent=4)

            self.status.showMessage("ROIs saved to {}".format(self.fn_rois), 1000)

    def savekeyboard(self, key):
        """Saves the annotation

        Args:
            key (Qt.Key): the pressed key
        """
        modifiers = QApplication.keyboardModifiers()

        if key == Qt.Key.Key_S and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.save()



if __name__ == '__main__':
    print("Starting application")
    app = QApplication(sys.argv)
    print("QApplication created")
    m = Main()
    m.show()
    sys.exit(app.exec())

