# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'home.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from Custom_Widgets.Widgets import QCustomSlideMenu
from Custom_Widgets.Widgets import QCustomStackedWidget
from Custom_Widgets.Widgets import *
import cv2
import gxipy as gx
import time
import uuid
import subprocess
import wget
import shutil
import zipfile
import tarfile
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
import glob
import resources_rc

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        loadJsonStyle(self, self)
        self.timer_camera = QtCore.QTimer(self)
        
        self.images = [] 
        self.image_path = []
        self.slot_init()
        self.show()

        # if Ui_MainWindow.detect_fn is None:
        #     Ui_MainWindow.detect_fn = self.create_detect_fn()

        # if Ui_MainWindow.category_index is None:
        #     Ui_MainWindow.category_index = self.create_category_index()

        # EXPAND Center Menu WIDGET 
        self.settingsButton.clicked.connect(lambda: self.ui.centerMenuContainer.expandMenu())
        self.cameraSetupbutton.clicked.connect(lambda: self.ui.centerMenuContainer.expandMenu())
        self.helpButton.clicked.connect(lambda: self.ui.centerMenuContainer.expandMenu())

        # CLOSE Center Menu WIDGET 
        self.closeCenterMenuButton.clicked.connect(lambda: self.ui.centerMenuContainer.collapseMenu())

        # EXPAND RIGHT Menu WIDGET 
        self.moreMenuButton.clicked.connect(lambda: self.ui.rightMenuContainer.expandMenu())
        self.imagesListButton.clicked.connect(lambda: self.ui.rightMenuContainer.expandMenu())
        

        # CLOSE RIGHT Menu WIDGET 
        self.closeRightmenuButton.clicked.connect(lambda: self.ui.rightMenuContainer.collapseMenu())
        
        # CLOSE NOTIFICATIONS Menu WIDGET 
        self.closeNotificationsButton.clicked.connect(lambda: self.ui.popupNotificationContainer.collapseMenu())

        #CONNECTING BUTTONS OBJECT DETECTION
        self.openCameraButton.clicked.connect(lambda: self.collect_images_fuction())
        self.captureImageButton.clicked.connect(self.capture)
        self.labelImagesButton.clicked.connect(self.labelImg)
        self.downloadModelButton.clicked.connect(self.training_function)
        self.gentfrecordsButton.clicked.connect(self.label_map_creation)
        self.trainButton.clicked.connect(self.model_training)
        self.detectImageButton.clicked.connect(self.detect_from_an_image)
        self.detectRealTimeButton.clicked.connect(self.Detect_in_real_time_camera)

        #CONNECTING GX DEVICE
        self.device_manager = gx.DeviceManager()
        self.device_manager.update_device_list()
        self.device = None


        #CONNECTING BUTTONS CLASSIFIER
        #self.openCamerabtnclassifier.clicked.connect(lambda: self.collect_images_fuction_classifier())


        

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
        #self.timer_camera.timeout.connect(self.show_camera_classifier)
    def show_camera(self):
        if self.device is None:
            return
        
        try:
            raw_image = self.device.data_stream[0].get_image()
            if raw_image is None:
                return
            
            color_image = raw_image.convert("RGB")
            show = color_image.get_numpy_array()
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            height, width, _ = show.shape
            show_image = QtGui.QImage(show, width, height, QtGui.QImage.Format_RGB888)
            self.label_17.setPixmap(QtGui.QPixmap.fromImage(show_image))
        except Exception as e:
            print(f"Error: {e}")
        # flag, self.imageopened = self.cap.read()
        # screen_width, screen_height = QDesktopWidget().screenGeometry().width(), QDesktopWidget().screenGeometry().height()
        # #resize the image to fit the screen width, and keep the aspect ratio
        # width_ratio = screen_width / self.imageopened.shape[1]
        # height_ratio = screen_height / self.imageopened.shape[0]
        # ratio = min(width_ratio, height_ratio)
        # show = cv2.resize(self.imageopened, (0,0), fx=ratio, fy=ratio)
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # self.showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showImage))

    def collect_images_fuction(self):


        if self.timer_camera.isActive() == False:
            device_info_list = self.device_manager.update_device_list()
            if len(device_info_list) == 0:
                msg = QtWidgets.QMessageBox.warning(self, "Warning", "Please check if the camera is properly connected to the computer",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                return
      
            self.device = self.device_manager.open_device_by_index(1)
            self.device.stream_on()
            self.timer_camera.start(30)
            self.openCameraButton.setText("Camera Opened")
        else:
            self.timer_camera.stop()
            self.device.stream_off()
    global IMG_PATH
    global labelsFolder
    global number_imgs
    global LABELIMG_PATH

    IMG_PATH = os.path.join('.\data\\workspace\\images\\')
    labelsFolder = ['OK', 'NG']
    number_imgs = 3
    LABELIMG_PATH = os.path.join('.\data\\', 'labelimg')

    def capture(self):
        if not os.path.exists(IMG_PATH):
            if os.name == 'posix':
                os.makedirs -p ({IMG_PATH})
            if os.name == 'nt':
                os.makedirs(IMG_PATH)
        for label in labelsFolder:
            path = os.path.join(IMG_PATH, label)
            if not os.path.exists(path):
                os.makedirs(path)

        for label in labelsFolder:
            pass
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            for imgnum in range(number_imgs):
                print('Collecting image {}'.format(imgnum))
                try:
                    if self.device is None:
                        self.device_manager.update_device_list()
                        dev_num, self.device = self.device_manager.open_device_by_index(1)
                        self.device.stream_on()
                    
                    #get raw data
                    raw_image = self.device.data_stream[0].get_image()
                    #convert to numpy array
                    image_data = raw_image.get_numpy_array()
                    # save image and add to the list of captured images
                    imgname = os.path.join(IMG_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                    cv2.imwrite(imgname, image_data)
                    time.sleep(2)
                    #convert to Qpixmap and add to the lsit of captured images
                    show = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                    qImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                    qPixmap = QtGui.QPixmap.fromImage(qImage)
                    self.images.append((qPixmap, label))
                except Exception as e:
                    print("Error capturing image: {}".format(str(e)))
                # if ret:
                #     imgname = os.path.join(IMG_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                #     cv2.imwrite(imgname, frame)
                #     time.sleep(2)
                #     show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     qImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                #     qPixmap = QtGui.QPixmap.fromImage(qImage)
                #     self.images.append((qPixmap, label))
              
        self.update_stack_of_images()




    def update_stack_of_images(self):

        stack = QVBoxLayout()
        for image, label_text in self.images:
            pass
            label = QLabel()
            label.setPixmap(image)
            label_text = QLabel(label_text)
            stack.addWidget(label)
            stack.addWidget(label_text)
        self.label_8.setLayout(stack)

    def labelImg(self):
        
        if not os.path.exists(LABELIMG_PATH):
             pass
             os.makedirs(LABELIMG_PATH)
             os.system(f"git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}")
        
        if os.name == 'posix':
             pass
             os.system("make qt5py3")
        if os.name == 'nt':
             pass
             os.system(f"cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc")

        process = subprocess.Popen(f"cd {LABELIMG_PATH} && python labelImg.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            print(f"Error: {err}")
        self.label_17.setText(out.decode("utf-8"))



    global CUSTOM_MODEL_NAME
    global PRETRAINED_MODEL_NAME
    global PRETRAINED_MODEL_URL
    global TF_RECORD_SCRIPT_NAME
    global LABEL_MAP_NAME
    global labels

    CUSTOM_MODEL_NAME = 'ssd_resnet1024_to_be_used100k' 
    PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt' 
    protoc = 'protoc-3.15.6-win64.zip' 
    unzip_protoc = 'data/protoc/tar -xf protoc-3.15.6-win64.zip'
    labels = [{'name':'OK', 'id':1}, {'name':'NG', 'id':2}]


    def training_function(self):

        paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
                'APIMODEL_PATH': os.path.join('data','models'),
                'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('data', 'workspace','images'),
                'MODEL_PATH': os.path.join('data', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('data','protoc')
                }
        files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
                }
        for path in paths.values():
            if not os.path.exists(path):
                if os.name == 'posix':
                    os.makedirs -p({path})
                if os.name == 'nt':
                    os.makedirs(path)
        if os.name=='nt':
             os.system(f"pip install wget")
        if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
             os.system(f"git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}")
    
        if os.name == 'nt':
             pass
             url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"

             VERIFICATION_SCRIPT2= subprocess.Popen('cd data/models/research/object_detection/builders/ && python model_builder_tf2_test.py', shell=True)
             print(VERIFICATION_SCRIPT2)

        if os.name =='posix':
            pass
            wget.download(PRETRAINED_MODEL_URL)
            pretrained_model_zip = zipfile.ZipFile(PRETRAINED_MODEL_NAME + '.tar.gz')
            pretrained_model_zip.extractall(paths['PRETRAINED_MODEL_PATH'])

        if os.name == 'nt':
            pass
            wget.download(PRETRAINED_MODEL_URL)
            pretrained_model_file = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz')
            os.rename(PRETRAINED_MODEL_NAME + '.tar.gz', pretrained_model_file)
            with tarfile.open(pretrained_model_file, "r:gz") as tar:
                pass
                tar.extractall(paths['PRETRAINED_MODEL_PATH'])
        #      zip_file1 = "protoc-3.15.6-win64.zip"
        #      research_path = os.path.join("data", "models", "research")
        #      setup_path = os.path.join("data","models", "research","object_detection", "packages", "tf2", "setup.py")
        #      destination = os.path.join(research_path, "setup.py")
        #      if os.name == 'nt':
                  
        #           try:
        #                if not os.path.exists(os.path.join(paths['PROTOC_PATH'], zip_file1)):
        #                     pass
        #                     wget.download(url)
        #                     shutil.move(zip_file1, paths['PROTOC_PATH'])
                       
        #           except FileNotFoundError:
        #                pass
        #                print(f"Error: {zip_file1} not found.")
        #           except subprocess.CalledProcessError as error:
        #                pass
        #                print(f"Error: {error}")
        #           else:
                       
        #                with zipfile.ZipFile(os.path.join(paths['PROTOC_PATH'], zip_file1), 'r') as zip_ref:
                            
        #                     zip_ref.extractall(paths['PROTOC_PATH'])
        #                     os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))
        #                try:
        #                     shutil.copy2(setup_path, destination)
        #                     subprocess.run(['cd', research_path, '&&', 'protoc', 'object_detection/protos/*.proto', '--python_out=.', '&&', 'python', 'setup.py', 'build', '&&', 'python', 'setup.py', 'install'], check=True, shell=True)
                            
                       
        #                     #subprocess.run(['cd', 'data/models/research', '&&', 'protoc', 'object_detection/protos/*.proto', '--python_out=.', '&&', 'copy', 'object_detection\\packages\\tf2\\setup.py', 'setup.py', '&&', 'python', 'setup.py', 'build', '&&', 'python', 'setup.py', 'install'], check=True)
        #                     subprocess.run(['cd', 'data/models/research/slim', '&&', 'pip', 'install', '-e', '.'], check=True)
        #                except subprocess.CalledProcessError as error:
        #                     pass
        #                     print(f"Error: {error}")
        #                else:
        #                     pass
        #                     print("Code ran successfully.")

        





    def label_map_creation(self):
        paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
                'APIMODEL_PATH': os.path.join('data','models'),
                'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('data', 'workspace','images'),
                'MODEL_PATH': os.path.join('data', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('data','protoc')
                }
        files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
                }

        labels = [{'name':'NG', 'id':1}, {'name':'OK', 'id':2}]

        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

        if not os.path.exists(files['TF_RECORD_SCRIPT']):
            os.system(f'git clone https://github.com/nicknochnack/GenerateTFRecord.git {paths["SCRIPTS_PATH"]}')

        
        train_command = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}"
        test_command = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}"

        subprocess.call(train_command, shell=True)
        subprocess.call(test_command, shell=True)

        if os.name == 'posix':
            subprocess.run(["cp", os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), os.path.join(paths['CHECKPOINT_PATH'])])
        elif os.name == 'nt':
            subprocess.run(["copy", os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), os.path.join(paths['CHECKPOINT_PATH'])], shell=True)

        print("Pipeline Config Successful")
        config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
                                                                                                                                                                                                                              
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config)
        pipeline_config.model.ssd.num_classes = len(labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
            
                                                                                                                                                                                                                                
            f.write(config_text)
            print(config_text)  


    def model_training(self):
        paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
                'APIMODEL_PATH': os.path.join('data','models'),
                'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('data', 'workspace','images'),
                'MODEL_PATH': os.path.join('data', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('data','protoc')
                }
        files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
                }
        TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        #command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=110000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
        #Evaluate the model
        command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
        subprocess.run(command, shell=True)
        

#     def create_detect_fn(self):
#         paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
#                 'APIMODEL_PATH': os.path.join('data','models'),
#                 'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
#                 'IMAGE_PATH': os.path.join('data', 'workspace','images'),
#                 'MODEL_PATH': os.path.join('data', 'workspace','models'),
#                 'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
#                 'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
#                 'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
#                 'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
#                 'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
#                 'PROTOC_PATH':os.path.join('data','protoc')
#                 }
#         files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
#                 'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
#                 'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
#                 }
#         # Load pipeline config and build a detection model
#         configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
#         detection_model = model_builder.build(model_config=configs['model'], is_training=False)

#         # Restore checkpoint
#         ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
#         ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-31')).expect_partial()
#         @tf.function
#         def detect_fn(image):
#             image, shapes = detection_model.preprocess(image)
#             prediction_dict = detection_model.predict(image, shapes)
#             detections = detection_model.postprocess(prediction_dict, shapes)
#             return detections

#         return detect_fn
    
#     def create_category_index(self):
#         paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
#                 'APIMODEL_PATH': os.path.join('data','models'),
#                 'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
#                 'IMAGE_PATH': os.path.join('data', 'workspace','images'),
#                 'MODEL_PATH': os.path.join('data', 'workspace','models'),
#                 'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
#                 'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
#                 'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
#                 'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
#                 'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
#                 'PROTOC_PATH':os.path.join('data','protoc')
#                 }
#         files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
#                 'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
#                 'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
#                 }
#         return label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    
#     def detect_from_an_image(self):
#         # Get a list of all images in the test folder
#         self.images = [cv2.imread(file) for file in glob.glob('./test/*.jpg')]

#         # Display the stack of images
#         self.display_stack()

#         # Connect the clicked signal of the QLabel to the slot function
#         self.label_8.clicked.connect(self.detect_image)

#     def display_image(self, image):
#         height, width, channel = image.shape
#         bytes_per_line = 3 * width
#         q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
#         self.label_8.setPixmap(QPixmap.fromImage(q_image))

#     def display_stack(self):
#         # Display the first image in the stack
#         self.display_image(self.images[0])

#     def detect_image(self):
#         # Get the currently displayed image
#         self.image_np = np.array(self.imageopened)

#         self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_np, 0), dtype=tf.float32)
#         self.detections = Ui_MainWindow.detect_fn(self.input_tensor)

#         self.num_detections = int(self.detections.pop('num_detections'))
#         self.detections = {key: value[0, :self.num_detections].numpy()
#                     for key, value in self.detections.items()}
#         self.detections['num_detections'] = self.num_detections
#         # detection_classes should be ints.
#         self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)
#         label_id_offset = 1

#         self.image_np_with_detections = self.image_np.copy()
#         viz_utils.visualize_boxes_and_labels_on_image_array(
#                     self.image_np_with_detections,
#                     self.detections['detection_boxes'],
#                     self.detections['detection_classes']+label_id_offset,
#                     self.detections['detection_scores'],
#                     Ui_MainWindow.category_index,
#                     use_normalized_coordinates=True,
#                     max_boxes_to_draw=10,
#                     min_score_thresh=.5,
#                     agnostic_mode=False)
#         showscreen = cv2.resize(self.image_np_with_detections, (640, 480 ))
#         showscreen = cv2.cvtColor(self.image_np_with_detections, cv2.COLOR_BGR2RGB)
#         self.showDetections = QtGui.QImage(showscreen.data, showscreen.shape[1], showscreen.shape[0], QtGui.QImage.Format_RGB888)
#         self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
#         self.label_17.setScaledContents(True)
    


#     def detect_from_an_image(self):
#         pass
#         # Load pipeline config and build a detection model
#         configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
#         detection_model = model_builder.build(model_config=configs['model'], is_training=False)

#         # Restore checkpoint
#         ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
#         ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-43')).expect_partial()

#         @tf.function
#         def detect_fn(image):
#             pass
        
#             image, shapes = detection_model.preprocess(image)
#             prediction_dict = detection_model.predict(image, shapes)
#             detections = detection_model.postprocess(prediction_dict, shapes)
#             return detections

#         category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

#         # Get a list of all images in the test folder
#         self.images = [cv2.imread(file) for file in glob.glob('./test/*.jpg')]

#         # Display the stack of images
#         self.display_stack()

#         # Connect the clicked signal of the QLabel to the slot function
#         self.label_8.clicked.connect(self.detect_image)

#     def display_stack(self):
#         pass
    
#         # Display the first image in the stack
#         self.display_image(self.images[0])

#     def detect_image(self):
#         pass
#         # Get the currently displayed image
#         self.image_np = np.array(self.imageopened)

#         self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_np, 0), dtype=tf.float32)
#         self.detections = detect_fn(self.input_tensor)

#         self.num_detections = int(self.detections.pop('num_detections'))
#         self.detections = {key: value[0, :self.num_detections].numpy()
#                     for key, value in self.detections.items()}
#         self.detections['num_detections'] = self.num_detections

#         # detection_classes should be ints.
#         self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

#         label_id_offset = 1
#         self.image_np_with_detections = self.image_np.copy()

#         viz_utils.visualize_boxes_and_labels_on_image_array(
#                     self.image_np_with_detections,
#                     self.detections['detection_boxes'],
#                     self.detections['detection_classes']+label_id_offset,
#                     self.detections['detection_scores'],
#                     category_index,
#                     use_normalized_coordinates=True,
#                     max_boxes_to_draw=10,
#                     min_score_thresh=.5,
#                     agnostic_mode=False)
#         showscreen = cv2.resize(self.image_np_with_detections, (640, 480 ))
#         showscreen = cv2.cvtColor(self.image_np_with_detections, cv2.COLOR_BGR2RGB)

    def detect_from_an_image(self): 
        paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
                'APIMODEL_PATH': os.path.join('data','models'),
                'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('data', 'workspace','images'),
                'MODEL_PATH': os.path.join('data', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('data','protoc')
                }
        files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
                }
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-111')).expect_partial()

        @tf.function
        def detect_fn(image):
            pass
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        self.img,ftype = QFileDialog.getOpenFileName(self, "Open File", "./", "Image Files(*.png)")
            
        self.imageopened=cv2.imread(r''.join(self.img))
        #self.image.setPixmap(QPixmap(img))
        #self.image.setScaledContents(True)
        self.image_np = np.array(self.imageopened)

        self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_np, 0), dtype=tf.float32)
        self.detections = detect_fn(self.input_tensor)

        self.num_detections = int(self.detections.pop('num_detections'))
        self.detections = {key: value[0, :self.num_detections].numpy()
                    for key, value in self.detections.items()}
        self.detections['num_detections'] = self.num_detections

        # detection_classes should be ints.
        self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        self.image_np_with_detections = self.image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    self.image_np_with_detections,
                    self.detections['detection_boxes'],
                    self.detections['detection_classes']+label_id_offset,
                    self.detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=20,
                    min_score_thresh=.5,
                    agnostic_mode=False)
        
        showscreen = cv2.resize(self.image_np_with_detections, (640, 480 ))
        showscreen = cv2.cvtColor(self.image_np_with_detections, cv2.COLOR_BGR2RGB)
        showscreen1 = cv2.resize(self.imageopened, (640, 480 ))
        showscreen1 = cv2.cvtColor(self.imageopened, cv2.COLOR_BGR2RGB)
        #show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self.showDetections = QtGui.QImage(showscreen.data, showscreen.shape[1], showscreen.shape[0], QtGui.QImage.Format_RGB888)
        self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
        self.label_17.setScaledContents(True)
        
        #frame stil Image 2
        # qt_img = QtGui.QImage(showscreen1.data.tobytes(),showscreen1.shape[1], showscreen1.shape[0], QtGui.QImage.Format_RGB888)
        # #new_img = qt_img.scaled(self.pic.width(), self.pic.height())
        # self.image.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        # self.image.setScaledContents(True)



    def Detect_in_real_time_camera(self):
        paths = {  'WORKSPACE_PATH': os.path.join('data', 'workspace'), 'SCRIPTS_PATH': os.path.join('data','scripts'),
                'APIMODEL_PATH': os.path.join('data','models'),
                'ANNOTATION_PATH': os.path.join('data', 'workspace','annotations'),
                'IMAGE_PATH': os.path.join('data', 'workspace','images'),
                'MODEL_PATH': os.path.join('data', 'workspace','models'),
                'PRETRAINED_MODEL_PATH': os.path.join('data', 'workspace','pre-trained-models'),
                'CHECKPOINT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME), 
                'OUTPUT_PATH': os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
                'TFJS_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
                'TFLITE_PATH':os.path.join('data', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
                'PROTOC_PATH':os.path.join('data','protoc')
                }
        files = {'PIPELINE_CONFIG':os.path.join('data', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
                'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
                'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-111')).expect_partial()

        @tf.function
        def detect_fn(image):
            pass
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        
        self.device_manager.update_device_list()
        self.device = self.device_manager.open_device_by_index(1)

        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        while True:
            
            try:

                if self.device is None:
                        print("No U3V device is found")
                        exit()
                self.device.stream_on()
                # Get the latest image from the camera
                self.raw_image = self.device.data_stream[0].get_image()
                self.raw_image = self.raw_image.convert("RGB")
                self.image_data = self.raw_image.get_numpy_array()
                if self.raw_image is None:
                    raise ValueError("No Image received from camera")
                self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_data, 0), dtype=tf.float32)
                self.detections = detect_fn(self.input_tensor)
                self.num_detections = int(self.detections.pop('num_detections'))
                self.detections = {key: value[0, :self.num_detections].numpy()
                                   for key, value in self.detections.items()}
                self.detections['num_detections'] = self.num_detections
                # detection_classes should be ints.
                self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)
                label_id_offset = 1
                self.image_np_with_detections = self.image_data.copy()
                viz_utils.visualize_boxes_and_labels_on_image_array(
                            self.image_np_with_detections,
                            self.detections['detection_boxes'],
                            self.detections['detection_classes']+label_id_offset,
                            self.detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            min_score_thresh=.3,
                            agnostic_mode=False)
                
                #self.render_detections = cv2.imshow('object detection',  cv2.resize(self.image_np_with_detections, (width, height)))
                showscreen = cv2.resize(self.image_np_with_detections, (540, 460))
                showscreen = cv2.cvtColor(self.image_np_with_detections, cv2.COLOR_BGR2RGB)
                #show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                self.showDetections = QtGui.QImage(showscreen.data, showscreen.shape[1], showscreen.shape[0], QtGui.QImage.Format_RGB888)
                self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
                self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
                #self.showDetections = QtGui.QImage(render_to_screen.data, render_to_screen.shape[1], render_to_screen.shape[0], QtGui.QImage.Format_RGB888)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"error occured {e}")
        self.device.stream_off()
        cv2.destroyAllWindows()
             





        # try:     
        #     if  self.device is None:
        #                 print("No U3V device found")
        #                 exit()
        #     self.device.stream_on()

        #     # Get the latest image from the camera
        #     self.raw_image = self.device.data_stream[0].get_image()
        #     self.raw_image = self.raw_image.convert("RGB")
        #     self.image_data = self.raw_image.get_numpy_array()
        #     if self.raw_image is None:
        #         raise ValueError("No Image received from camera")
         
        
        #     self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_data, 0), dtype=tf.float32)
        #     self.detections = detect_fn(self.input_tensor)
        
        #     self.num_detections = int(self.detections.pop('num_detections'))
        #     self.detections = {key: value[0, :self.num_detections].numpy()
        #                         for key, value in self.detections.items()}
        #     self.detections['num_detections'] = self.num_detections

        #     # detection_classes should be ints.
        #     self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

        #     label_id_offset = 1
        #     self.image_np_with_detections = self.image_data.copy()

        #     viz_utils.visualize_boxes_and_labels_on_image_array(
        #                     self.image_np_with_detections,
        #                     self.detections['detection_boxes'],
        #                     self.detections['detection_classes']+label_id_offset,
        #                     self.detections['detection_scores'],
        #                     category_index,
        #                     use_normalized_coordinates=True,
        #                     max_boxes_to_draw=5,
        #                     min_score_thresh=.3,
        #                     agnostic_mode=False)

        #     #self.render_detections = cv2.imshow('object detection',  cv2.resize(self.image_np_with_detections, (width, height)))
        #     showscreen = cv2.resize(self.image_np_with_detections, (540, 460))
        #     showscreen = cv2.cvtColor(self.image_np_with_detections, cv2.COLOR_BGR2RGB)
        #     #show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        #     self.showDetections = QtGui.QImage(showscreen.data, showscreen.shape[1], showscreen.shape[0], QtGui.QImage.Format_RGB888)
        #     self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
        #     self.label_17.setPixmap(QtGui.QPixmap.fromImage(self.showDetections))
        #     #self.showDetections = QtGui.QImage(render_to_screen.data, render_to_screen.shape[1], render_to_screen.shape[0], QtGui.QImage.Format_RGB888)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #             pass
                    
        #         
            
        # except Exception as e:
        #     print(f"Error occured {e}")
        # self.device.stream_off()
        # cv2.destroyAllWindows()
        
                    
      
                    
                    

        
#
#
#
#
#
#
################################################################################
## Form generated from reading UI file 'home.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## MODULE: CLASSIFIER!!! 
################################################################################
#
#
#
#
#
    # def show_camera_classifier(self):
            
    #     flag, self.imageopenedclassifier = self.cap.read()
    #     show = cv2.resize(self.imageopenedclassifier, (640, 480))
    #     show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
    #     self.showImageclassifier = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
    #     self.label_18.setPixmap(QtGui.QPixmap.fromImage(self.showImageclassifier))

    # def collect_images_fuction_classifier(self):
    #     if self.timer_camera.isActive() == False:
    #         flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)
    #         if flag == False:
    #             msg = QtWidgets.QMessageBox.warning(self, "Warning", "Please check if the camera is properly connected to the computer",
    #                                                 buttons=QtWidgets.QMessageBox.Ok,
    #                                                 defaultButton=QtWidgets.QMessageBox.Ok)
    #         else:
    #             self.timer_camera.start(30)
    #             self.openCamerabtnclassifier.setText("Camera Opened")
    #     else:
    #         self.timer_camera.stop()
    #         self.cap.release()
    #         self.label_18.clear()
    #         self.openCamerabtnclassifier.setText("Camera Closed")



    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1265, 899)
        MainWindow.setStyleSheet(u"*{\n"
"	border:none;\n"
"	background-color:transparent;\n"
"	background:transparent;\n"
"	padding:0;\n"
"	margin:0;\n"
"	color:#fff;\n"
"\n"
"\n"
"}\n"
"\n"
"#centralwidget{\n"
"	background-color:#1f232a;\n"
"\n"
"}\n"
"#leftMenuSubContainer{\n"
"	background-color:#16191d;\n"
"\n"
"\n"
"}\n"
"QPushButton{\n"
"	text-align:left;\n"
"	padding: 5px 10px\n"
"\n"
"}\n"
"#centerMenuSubContainer,#rightMenuSubContainer{\n"
"	background-color:#2c313c;\n"
"\n"
"}\n"
"#frame_4, #frame_8,#popupNotificationSubContainer{\n"
"	background-color:#16191d;\n"
"	border-radius:10px;\n"
"}\n"
"#headerContainer, #footerContainer{\n"
"	background-color:#2c313c;\n"
"\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.leftMenuContainer = QCustomSlideMenu(self.centralwidget)
        self.leftMenuContainer.setObjectName(u"leftMenuContainer")
        self.leftMenuContainer.setMinimumSize(QSize(0, 0))
        self.leftMenuContainer.setMaximumSize(QSize(50, 16777215))
        self.verticalLayout = QVBoxLayout(self.leftMenuContainer)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.leftMenuSubContainer = QWidget(self.leftMenuContainer)
        self.leftMenuSubContainer.setObjectName(u"leftMenuSubContainer")
        self.verticalLayout_2 = QVBoxLayout(self.leftMenuSubContainer)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 0, 0, 0)
        self.frame = QFrame(self.leftMenuSubContainer)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.menuButton = QPushButton(self.frame)
        self.menuButton.setObjectName(u"menuButton")
        self.menuButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon = QIcon()
        icon.addFile(u":/icons/feather/align-justify.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.menuButton.setIcon(icon)
        self.menuButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.menuButton)


        self.verticalLayout_2.addWidget(self.frame, 0, Qt.AlignTop)

        self.frame_2 = QFrame(self.leftMenuSubContainer)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 10, 0, 10)
        self.homebutton = QPushButton(self.frame_2)
        self.homebutton.setObjectName(u"homebutton")
        self.homebutton.setCursor(QCursor(Qt.PointingHandCursor))
        self.homebutton.setStyleSheet(u"background-color:#1f232a;")
        icon1 = QIcon()
        icon1.addFile(u":/icons/feather/home.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.homebutton.setIcon(icon1)
        self.homebutton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.homebutton)

        self.dataAnalysisbutton = QPushButton(self.frame_2)
        self.dataAnalysisbutton.setObjectName(u"dataAnalysisbutton")
        self.dataAnalysisbutton.setCursor(QCursor(Qt.PointingHandCursor))
        icon2 = QIcon()
        icon2.addFile(u":/icons/feather/list.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.dataAnalysisbutton.setIcon(icon2)
        self.dataAnalysisbutton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.dataAnalysisbutton)

        self.cpu_gpubutton = QPushButton(self.frame_2)
        self.cpu_gpubutton.setObjectName(u"cpu_gpubutton")
        self.cpu_gpubutton.setCursor(QCursor(Qt.PointingHandCursor))
        icon3 = QIcon()
        icon3.addFile(u":/icons/feather/cpu.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.cpu_gpubutton.setIcon(icon3)
        self.cpu_gpubutton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.cpu_gpubutton)

        self.ObjectDetectionButton = QPushButton(self.frame_2)
        self.ObjectDetectionButton.setObjectName(u"ObjectDetectionButton")
        self.ObjectDetectionButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon4 = QIcon()
        icon4.addFile(u":/icons/feather/figma.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.ObjectDetectionButton.setIcon(icon4)
        self.ObjectDetectionButton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.ObjectDetectionButton)

        self.classificationButton = QPushButton(self.frame_2)
        self.classificationButton.setObjectName(u"classificationButton")
        self.classificationButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon5 = QIcon()
        icon5.addFile(u":/icons/feather/copy.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.classificationButton.setIcon(icon5)
        self.classificationButton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.classificationButton)

        self.QrCodeButton = QPushButton(self.frame_2)
        self.QrCodeButton.setObjectName(u"QrCodeButton")
        self.QrCodeButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon6 = QIcon()
        icon6.addFile(u":/icons/feather/columns.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.QrCodeButton.setIcon(icon6)
        self.QrCodeButton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.QrCodeButton)

        self.codeScanGunButton = QPushButton(self.frame_2)
        self.codeScanGunButton.setObjectName(u"codeScanGunButton")
        self.codeScanGunButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon7 = QIcon()
        icon7.addFile(u":/icons/feather/aperture.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.codeScanGunButton.setIcon(icon7)
        self.codeScanGunButton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.codeScanGunButton)

        self.plcButton = QPushButton(self.frame_2)
        self.plcButton.setObjectName(u"plcButton")
        self.plcButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon8 = QIcon()
        icon8.addFile(u":/icons/feather/crosshair.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.plcButton.setIcon(icon8)
        self.plcButton.setIconSize(QSize(24, 24))

        self.verticalLayout_3.addWidget(self.plcButton)


        self.verticalLayout_2.addWidget(self.frame_2, 0, Qt.AlignTop)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.frame_3 = QFrame(self.leftMenuSubContainer)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 10, 0, 10)
        self.cameraSetupbutton = QPushButton(self.frame_3)
        self.cameraSetupbutton.setObjectName(u"cameraSetupbutton")
        self.cameraSetupbutton.setCursor(QCursor(Qt.PointingHandCursor))
        icon9 = QIcon()
        icon9.addFile(u":/icons/feather/camera.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.cameraSetupbutton.setIcon(icon9)
        self.cameraSetupbutton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.cameraSetupbutton)

        self.settingsButton = QPushButton(self.frame_3)
        self.settingsButton.setObjectName(u"settingsButton")
        self.settingsButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon10 = QIcon()
        icon10.addFile(u":/icons/feather/settings.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.settingsButton.setIcon(icon10)
        self.settingsButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.settingsButton)

        self.helpButton = QPushButton(self.frame_3)
        self.helpButton.setObjectName(u"helpButton")
        self.helpButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon11 = QIcon()
        icon11.addFile(u":/icons/feather/help-circle.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.helpButton.setIcon(icon11)
        self.helpButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.helpButton)

        self.logsButton = QPushButton(self.frame_3)
        self.logsButton.setObjectName(u"logsButton")
        self.logsButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon12 = QIcon()
        icon12.addFile(u":/icons/feather/umbrella.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.logsButton.setIcon(icon12)
        self.logsButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.logsButton)


        self.verticalLayout_2.addWidget(self.frame_3, 0, Qt.AlignBottom)


        self.verticalLayout.addWidget(self.leftMenuSubContainer)


        self.horizontalLayout.addWidget(self.leftMenuContainer, 0, Qt.AlignLeft)

        self.centerMenuContainer = QCustomSlideMenu(self.centralwidget)
        self.centerMenuContainer.setObjectName(u"centerMenuContainer")
        self.centerMenuContainer.setMinimumSize(QSize(200, 0))
        self.verticalLayout_5 = QVBoxLayout(self.centerMenuContainer)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.centerMenuSubContainer = QWidget(self.centerMenuContainer)
        self.centerMenuSubContainer.setObjectName(u"centerMenuSubContainer")
        self.centerMenuSubContainer.setMinimumSize(QSize(200, 0))
        self.verticalLayout_6 = QVBoxLayout(self.centerMenuSubContainer)
        self.verticalLayout_6.setSpacing(5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(5, 5, 5, 5)
        self.frame_4 = QFrame(self.centerMenuSubContainer)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(self.frame_4)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label)

        self.closeCenterMenuButton = QPushButton(self.frame_4)
        self.closeCenterMenuButton.setObjectName(u"closeCenterMenuButton")
        self.closeCenterMenuButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon13 = QIcon()
        icon13.addFile(u":/icons/feather/x-circle.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.closeCenterMenuButton.setIcon(icon13)
        self.closeCenterMenuButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_3.addWidget(self.closeCenterMenuButton, 0, Qt.AlignRight)


        self.verticalLayout_6.addWidget(self.frame_4, 0, Qt.AlignTop)

        self.centerMenuPages = QCustomStackedWidget(self.centerMenuSubContainer)
        self.centerMenuPages.setObjectName(u"centerMenuPages")
        self.pageCamConfig = QWidget()
        self.pageCamConfig.setObjectName(u"pageCamConfig")
        self.verticalLayout_7 = QVBoxLayout(self.pageCamConfig)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.label_2 = QLabel(self.pageCamConfig)
        self.label_2.setObjectName(u"label_2")
        font = QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_2)

        self.centerMenuPages.addWidget(self.pageCamConfig)
        self.pageLogs = QWidget()
        self.pageLogs.setObjectName(u"pageLogs")
        self.verticalLayout_21 = QVBoxLayout(self.pageLogs)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.label_16 = QLabel(self.pageLogs)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setFont(font)
        self.label_16.setAlignment(Qt.AlignCenter)

        self.verticalLayout_21.addWidget(self.label_16)

        self.centerMenuPages.addWidget(self.pageLogs)
        self.pageSysSettings = QWidget()
        self.pageSysSettings.setObjectName(u"pageSysSettings")
        self.verticalLayout_8 = QVBoxLayout(self.pageSysSettings)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_3 = QLabel(self.pageSysSettings)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_3)

        self.centerMenuPages.addWidget(self.pageSysSettings)
        self.pageHelp = QWidget()
        self.pageHelp.setObjectName(u"pageHelp")
        self.verticalLayout_9 = QVBoxLayout(self.pageHelp)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_4 = QLabel(self.pageHelp)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.label_4)

        self.centerMenuPages.addWidget(self.pageHelp)

        self.verticalLayout_6.addWidget(self.centerMenuPages)


        self.verticalLayout_5.addWidget(self.centerMenuSubContainer, 0, Qt.AlignLeft)


        self.horizontalLayout.addWidget(self.centerMenuContainer)

        self.mainBodyContainer = QWidget(self.centralwidget)
        self.mainBodyContainer.setObjectName(u"mainBodyContainer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.mainBodyContainer.sizePolicy().hasHeightForWidth())
        self.mainBodyContainer.setSizePolicy(sizePolicy1)
        self.mainBodyContainer.setStyleSheet(u"")
        self.verticalLayout_10 = QVBoxLayout(self.mainBodyContainer)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.headerContainer = QWidget(self.mainBodyContainer)
        self.headerContainer.setObjectName(u"headerContainer")
        self.horizontalLayout_5 = QHBoxLayout(self.headerContainer)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.frame_5 = QFrame(self.headerContainer)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.frame_5)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMaximumSize(QSize(40, 40))
        self.label_5.setPixmap(QPixmap(u":/images/runmo.png"))
        self.label_5.setScaledContents(True)

        self.horizontalLayout_7.addWidget(self.label_5)

        self.label_6 = QLabel(self.frame_5)
        self.label_6.setObjectName(u"label_6")
        font1 = QFont()
        font1.setFamily(u"Bahnschrift")
        font1.setPointSize(12)
        font1.setBold(True)
        font1.setItalic(False)
        font1.setWeight(75)
        self.label_6.setFont(font1)

        self.horizontalLayout_7.addWidget(self.label_6)


        self.horizontalLayout_5.addWidget(self.frame_5, 0, Qt.AlignLeft)

        self.frame_6 = QFrame(self.headerContainer)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.zoomInButton = QPushButton(self.frame_6)
        self.zoomInButton.setObjectName(u"zoomInButton")
        self.zoomInButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon14 = QIcon()
        icon14.addFile(u":/icons/feather/zoom-in.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.zoomInButton.setIcon(icon14)
        self.zoomInButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_6.addWidget(self.zoomInButton)

        self.zoomOutButton = QPushButton(self.frame_6)
        self.zoomOutButton.setObjectName(u"zoomOutButton")
        self.zoomOutButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon15 = QIcon()
        icon15.addFile(u":/icons/feather/zoom-out.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.zoomOutButton.setIcon(icon15)
        self.zoomOutButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_6.addWidget(self.zoomOutButton)

        self.debugMessagesButton = QPushButton(self.frame_6)
        self.debugMessagesButton.setObjectName(u"debugMessagesButton")
        icon16 = QIcon()
        icon16.addFile(u":/icons/feather/tool.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.debugMessagesButton.setIcon(icon16)
        self.debugMessagesButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_6.addWidget(self.debugMessagesButton)

        self.moreMenuButton = QPushButton(self.frame_6)
        self.moreMenuButton.setObjectName(u"moreMenuButton")
        self.moreMenuButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon17 = QIcon()
        icon17.addFile(u":/icons/feather/more-horizontal.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.moreMenuButton.setIcon(icon17)
        self.moreMenuButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_6.addWidget(self.moreMenuButton)

        self.imagesListButton = QPushButton(self.frame_6)
        self.imagesListButton.setObjectName(u"imagesListButton")
        self.imagesListButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon18 = QIcon()
        icon18.addFile(u":/icons/feather/image.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.imagesListButton.setIcon(icon18)
        self.imagesListButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_6.addWidget(self.imagesListButton)


        self.horizontalLayout_5.addWidget(self.frame_6, 0, Qt.AlignHCenter)

        self.frame_7 = QFrame(self.headerContainer)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.minimizeButton = QPushButton(self.frame_7)
        self.minimizeButton.setObjectName(u"minimizeButton")
        icon19 = QIcon()
        icon19.addFile(u":/icons/feather/minus.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.minimizeButton.setIcon(icon19)

        self.horizontalLayout_4.addWidget(self.minimizeButton)

        self.restoreButton = QPushButton(self.frame_7)
        self.restoreButton.setObjectName(u"restoreButton")
        icon20 = QIcon()
        icon20.addFile(u":/icons/feather/square.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.restoreButton.setIcon(icon20)

        self.horizontalLayout_4.addWidget(self.restoreButton)

        self.closeButton = QPushButton(self.frame_7)
        self.closeButton.setObjectName(u"closeButton")
        icon21 = QIcon()
        icon21.addFile(u":/icons/feather/x.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.closeButton.setIcon(icon21)

        self.horizontalLayout_4.addWidget(self.closeButton)


        self.horizontalLayout_5.addWidget(self.frame_7, 0, Qt.AlignRight)


        self.verticalLayout_10.addWidget(self.headerContainer, 0, Qt.AlignTop)

        self.mainBodyContent = QWidget(self.mainBodyContainer)
        self.mainBodyContent.setObjectName(u"mainBodyContent")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.mainBodyContent.sizePolicy().hasHeightForWidth())
        self.mainBodyContent.setSizePolicy(sizePolicy2)
        self.mainBodyContent.setMinimumSize(QSize(719, 633))
        self.horizontalLayout_8 = QHBoxLayout(self.mainBodyContent)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.mainContentsContainer = QWidget(self.mainBodyContent)
        self.mainContentsContainer.setObjectName(u"mainContentsContainer")
        self.verticalLayout_15 = QVBoxLayout(self.mainContentsContainer)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.mainPages = QCustomStackedWidget(self.mainContentsContainer)
        self.mainPages.setObjectName(u"mainPages")
        self.pageHome = QWidget()
        self.pageHome.setObjectName(u"pageHome")
        self.verticalLayout_16 = QVBoxLayout(self.pageHome)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.label_10 = QLabel(self.pageHome)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font)
        self.label_10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_16.addWidget(self.label_10)

        self.mainPages.addWidget(self.pageHome)
        self.pageDataAnalysis = QWidget()
        self.pageDataAnalysis.setObjectName(u"pageDataAnalysis")
        self.verticalLayout_17 = QVBoxLayout(self.pageDataAnalysis)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.label_11 = QLabel(self.pageDataAnalysis)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)
        self.label_11.setAlignment(Qt.AlignCenter)

        self.verticalLayout_17.addWidget(self.label_11)

        self.mainPages.addWidget(self.pageDataAnalysis)
        self.pageCPUMetrics = QWidget()
        self.pageCPUMetrics.setObjectName(u"pageCPUMetrics")
        self.verticalLayout_18 = QVBoxLayout(self.pageCPUMetrics)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.label_12 = QLabel(self.pageCPUMetrics)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font)
        self.label_12.setAlignment(Qt.AlignCenter)

        self.verticalLayout_18.addWidget(self.label_12)

        self.mainPages.addWidget(self.pageCPUMetrics)
        self.pageObjDetection = QWidget()
        self.pageObjDetection.setObjectName(u"pageObjDetection")
        self.verticalLayout_22 = QVBoxLayout(self.pageObjDetection)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.frame_11 = QFrame(self.pageObjDetection)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.openCameraButton = QPushButton(self.frame_11)
        self.openCameraButton.setObjectName(u"openCameraButton")
        self.openCameraButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.openCameraButton.setAutoFillBackground(False)
        self.openCameraButton.setStyleSheet(u"background-color:#32576a;")
        icon22 = QIcon()
        icon22.addFile(u":/icons/feather/camera-off.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.openCameraButton.setIcon(icon22)
        self.openCameraButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.openCameraButton)

        self.labelImagesButton = QPushButton(self.frame_11)
        self.labelImagesButton.setObjectName(u"labelImagesButton")
        self.labelImagesButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.labelImagesButton.setStyleSheet(u"background-color:#32576a;")
        icon23 = QIcon()
        icon23.addFile(u":/icons/feather/edit-3.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.labelImagesButton.setIcon(icon23)
        self.labelImagesButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.labelImagesButton)

        self.captureImageButton = QPushButton(self.frame_11)
        self.captureImageButton.setObjectName(u"captureImageButton")
        self.captureImageButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.captureImageButton.setStyleSheet(u"background-color:#32576a;")
        self.captureImageButton.setIcon(icon9)
        self.captureImageButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.captureImageButton)

        self.trainButton = QPushButton(self.frame_11)
        self.trainButton.setObjectName(u"trainButton")
        self.trainButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.trainButton.setStyleSheet(u"background-color:#32576a;")
        icon24 = QIcon()
        icon24.addFile(u":/icons/feather/sliders.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.trainButton.setIcon(icon24)
        self.trainButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.trainButton)

        self.detectRealTimeButton = QPushButton(self.frame_11)
        self.detectRealTimeButton.setObjectName(u"detectRealTimeButton")
        self.detectRealTimeButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.detectRealTimeButton.setStyleSheet(u"background-color:#32576a;")
        icon25 = QIcon()
        icon25.addFile(u":/icons/feather/codepen.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.detectRealTimeButton.setIcon(icon25)
        self.detectRealTimeButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.detectRealTimeButton)

        self.detectImageButton = QPushButton(self.frame_11)
        self.detectImageButton.setObjectName(u"detectImageButton")
        self.detectImageButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.detectImageButton.setStyleSheet(u"background-color:#32576a;")
        self.detectImageButton.setIcon(icon25)
        self.detectImageButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.detectImageButton)

        self.downloadModelButton = QPushButton(self.frame_11)
        self.downloadModelButton.setObjectName(u"downloadModelButton")
        self.downloadModelButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.downloadModelButton.setStyleSheet(u"background-color:#32576a;")
        icon26 = QIcon()
        icon26.addFile(u":/icons/feather/download-cloud.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.downloadModelButton.setIcon(icon26)
        self.downloadModelButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.downloadModelButton)

        self.gentfrecordsButton = QPushButton(self.frame_11)
        self.gentfrecordsButton.setObjectName(u"gentfrecordsButton")
        self.gentfrecordsButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.gentfrecordsButton.setStyleSheet(u"background-color:#32576a;")
        icon27 = QIcon()
        icon27.addFile(u":/icons/feather/trello.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.gentfrecordsButton.setIcon(icon27)
        self.gentfrecordsButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_13.addWidget(self.gentfrecordsButton)


        self.verticalLayout_22.addWidget(self.frame_11, 0, Qt.AlignTop)

        self.label_17 = QLabel(self.pageObjDetection)
        self.label_17.setObjectName(u"label_17")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy3)
        self.label_17.setFont(font)
        self.label_17.setAutoFillBackground(True)
        self.label_17.setAlignment(Qt.AlignCenter)

        self.verticalLayout_22.addWidget(self.label_17)

        self.frame_13 = QFrame(self.pageObjDetection)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setFrameShape(QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_13)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")

        self.verticalLayout_22.addWidget(self.frame_13)

        self.mainPages.addWidget(self.pageObjDetection)
        self.pageClassification = QWidget()
        self.pageClassification.setObjectName(u"pageClassification")
        self.verticalLayout_23 = QVBoxLayout(self.pageClassification)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.widget = QWidget(self.pageClassification)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_15 = QHBoxLayout(self.widget)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.leftCameraHolderFrame = QFrame(self.widget)
        self.leftCameraHolderFrame.setObjectName(u"leftCameraHolderFrame")
        self.leftCameraHolderFrame.setFrameShape(QFrame.StyledPanel)
        self.leftCameraHolderFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.leftCameraHolderFrame)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.label_18 = QLabel(self.leftCameraHolderFrame)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setAutoFillBackground(True)
        self.label_18.setAlignment(Qt.AlignCenter)

        self.verticalLayout_27.addWidget(self.label_18)

        self.label_22 = QLabel(self.leftCameraHolderFrame)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setAutoFillBackground(False)
        self.label_22.setAlignment(Qt.AlignCenter)

        self.verticalLayout_27.addWidget(self.label_22)


        self.horizontalLayout_15.addWidget(self.leftCameraHolderFrame)

        self.line = QFrame(self.widget)
        self.line.setObjectName(u"line")
        self.line.setStyleSheet(u"background-color:rgb(85, 85, 255)")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_15.addWidget(self.line)

        self.rightFrameFunctions = QFrame(self.widget)
        self.rightFrameFunctions.setObjectName(u"rightFrameFunctions")
        self.rightFrameFunctions.setFrameShape(QFrame.StyledPanel)
        self.rightFrameFunctions.setFrameShadow(QFrame.Raised)
        self.verticalLayout_28 = QVBoxLayout(self.rightFrameFunctions)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.label_23 = QLabel(self.rightFrameFunctions)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setAutoFillBackground(True)
        self.label_23.setAlignment(Qt.AlignCenter)

        self.verticalLayout_28.addWidget(self.label_23)

        self.verticalSpacer_2 = QSpacerItem(0, 100, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.verticalLayout_28.addItem(self.verticalSpacer_2)

        self.groupBox = QGroupBox(self.rightFrameFunctions)
        self.groupBox.setObjectName(u"groupBox")
        self.splitter = QSplitter(self.groupBox)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setGeometry(QRect(30, 40, 186, 102))
        self.splitter.setOrientation(Qt.Vertical)
        self.openCamerabtnclassifier = QPushButton(self.splitter)
        self.openCamerabtnclassifier.setObjectName(u"openCamerabtnclassifier")
        self.openCamerabtnclassifier.setCursor(QCursor(Qt.PointingHandCursor))
        self.openCamerabtnclassifier.setStyleSheet(u"background-color:#32576a;")
        self.openCamerabtnclassifier.setIcon(icon22)
        self.openCamerabtnclassifier.setIconSize(QSize(24, 24))
        self.splitter.addWidget(self.openCamerabtnclassifier)
        self.captureBtnClassifier = QPushButton(self.splitter)
        self.captureBtnClassifier.setObjectName(u"captureBtnClassifier")
        self.captureBtnClassifier.setCursor(QCursor(Qt.PointingHandCursor))
        self.captureBtnClassifier.setStyleSheet(u"background-color:#32576a;")
        self.captureBtnClassifier.setIcon(icon9)
        self.captureBtnClassifier.setIconSize(QSize(24, 24))
        self.splitter.addWidget(self.captureBtnClassifier)
        self.plotGraphbuttonClassifier = QPushButton(self.splitter)
        self.plotGraphbuttonClassifier.setObjectName(u"plotGraphbuttonClassifier")
        self.plotGraphbuttonClassifier.setCursor(QCursor(Qt.PointingHandCursor))
        self.plotGraphbuttonClassifier.setStyleSheet(u"background-color:#32576a;")
        icon28 = QIcon()
        icon28.addFile(u":/icons/feather/map.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.plotGraphbuttonClassifier.setIcon(icon28)
        self.plotGraphbuttonClassifier.setIconSize(QSize(24, 24))
        self.splitter.addWidget(self.plotGraphbuttonClassifier)
        self.finalClassiybutton = QPushButton(self.splitter)
        self.finalClassiybutton.setObjectName(u"finalClassiybutton")
        self.finalClassiybutton.setCursor(QCursor(Qt.PointingHandCursor))
        self.finalClassiybutton.setStyleSheet(u"background-color:#32576a;")
        icon29 = QIcon()
        icon29.addFile(u":/icons/feather/activity.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.finalClassiybutton.setIcon(icon29)
        self.finalClassiybutton.setIconSize(QSize(24, 24))
        self.splitter.addWidget(self.finalClassiybutton)

        self.verticalLayout_28.addWidget(self.groupBox)


        self.horizontalLayout_15.addWidget(self.rightFrameFunctions)


        self.verticalLayout_23.addWidget(self.widget)

        self.mainPages.addWidget(self.pageClassification)
        self.pageOCRbarcode = QWidget()
        self.pageOCRbarcode.setObjectName(u"pageOCRbarcode")
        self.verticalLayout_24 = QVBoxLayout(self.pageOCRbarcode)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.label_19 = QLabel(self.pageOCRbarcode)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setFont(font)
        self.label_19.setAlignment(Qt.AlignCenter)

        self.verticalLayout_24.addWidget(self.label_19)

        self.mainPages.addWidget(self.pageOCRbarcode)
        self.pageCodeScanGun = QWidget()
        self.pageCodeScanGun.setObjectName(u"pageCodeScanGun")
        self.verticalLayout_25 = QVBoxLayout(self.pageCodeScanGun)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.label_20 = QLabel(self.pageCodeScanGun)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setFont(font)
        self.label_20.setAlignment(Qt.AlignCenter)

        self.verticalLayout_25.addWidget(self.label_20)

        self.mainPages.addWidget(self.pageCodeScanGun)
        self.pagePLC = QWidget()
        self.pagePLC.setObjectName(u"pagePLC")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.pagePLC.sizePolicy().hasHeightForWidth())
        self.pagePLC.setSizePolicy(sizePolicy4)
        self.verticalLayout_26 = QVBoxLayout(self.pagePLC)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.pagePlcSubContainer = QWidget(self.pagePLC)
        self.pagePlcSubContainer.setObjectName(u"pagePlcSubContainer")
        sizePolicy4.setHeightForWidth(self.pagePlcSubContainer.sizePolicy().hasHeightForWidth())
        self.pagePlcSubContainer.setSizePolicy(sizePolicy4)
        self.verticalLayout_29 = QVBoxLayout(self.pagePlcSubContainer)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.frame_12 = QFrame(self.pagePlcSubContainer)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Raised)
        self.splitter_3 = QSplitter(self.frame_12)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setGeometry(QRect(20, 60, 301, 31))
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.label_21 = QLabel(self.splitter_3)
        self.label_21.setObjectName(u"label_21")
        self.splitter_3.addWidget(self.label_21)
        self.comboBox = QComboBox(self.splitter_3)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setStyleSheet(u"background-color:#32576a;")
        self.splitter_3.addWidget(self.comboBox)
        self.splitter_2 = QSplitter(self.frame_12)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setGeometry(QRect(20, 30, 271, 31))
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.plc1Button = QRadioButton(self.splitter_2)
        self.plc1Button.setObjectName(u"plc1Button")
        self.splitter_2.addWidget(self.plc1Button)
        self.plc2Button = QRadioButton(self.splitter_2)
        self.plc2Button.setObjectName(u"plc2Button")
        self.splitter_2.addWidget(self.plc2Button)
        self.plc3Button = QRadioButton(self.splitter_2)
        self.plc3Button.setObjectName(u"plc3Button")
        self.splitter_2.addWidget(self.plc3Button)
        self.widget1 = QWidget(self.frame_12)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(20, 110, 451, 101))
        self.horizontalLayout_16 = QHBoxLayout(self.widget1)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setSizeConstraint(QLayout.SetMaximumSize)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.frame_15 = QFrame(self.widget1)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setFrameShape(QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Raised)
        self.verticalLayout_30 = QVBoxLayout(self.frame_15)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.splitter_4 = QSplitter(self.frame_15)
        self.splitter_4.setObjectName(u"splitter_4")
        self.splitter_4.setOrientation(Qt.Horizontal)
        self.label_24 = QLabel(self.splitter_4)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.splitter_4.addWidget(self.label_24)
        self.comboBox_2 = QComboBox(self.splitter_4)
        self.comboBox_2.addItem("")
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.comboBox_2.setStyleSheet(u"background-color:#32576a;")
        self.splitter_4.addWidget(self.comboBox_2)

        self.verticalLayout_30.addWidget(self.splitter_4)

        self.splitter_5 = QSplitter(self.frame_15)
        self.splitter_5.setObjectName(u"splitter_5")
        self.splitter_5.setOrientation(Qt.Horizontal)
        self.label_25 = QLabel(self.splitter_5)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.splitter_5.addWidget(self.label_25)
        self.comboBox_3 = QComboBox(self.splitter_5)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setStyleSheet(u"background-color:#32576a;")
        self.splitter_5.addWidget(self.comboBox_3)

        self.verticalLayout_30.addWidget(self.splitter_5)

        self.splitter_6 = QSplitter(self.frame_15)
        self.splitter_6.setObjectName(u"splitter_6")
        self.splitter_6.setOrientation(Qt.Horizontal)
        self.label_26 = QLabel(self.splitter_6)
        self.label_26.setObjectName(u"label_26")
        self.splitter_6.addWidget(self.label_26)
        self.comboBox_4 = QComboBox(self.splitter_6)
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.setObjectName(u"comboBox_4")
        self.comboBox_4.setStyleSheet(u"background-color:#32576a;")
        self.splitter_6.addWidget(self.comboBox_4)

        self.verticalLayout_30.addWidget(self.splitter_6)


        self.horizontalLayout_16.addWidget(self.frame_15)

        self.frame_16 = QFrame(self.widget1)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setFrameShape(QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Raised)
        self.verticalLayout_31 = QVBoxLayout(self.frame_16)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.splitter_7 = QSplitter(self.frame_16)
        self.splitter_7.setObjectName(u"splitter_7")
        self.splitter_7.setOrientation(Qt.Horizontal)
        self.label_27 = QLabel(self.splitter_7)
        self.label_27.setObjectName(u"label_27")
        font2 = QFont()
        font2.setStrikeOut(False)
        self.label_27.setFont(font2)
        self.splitter_7.addWidget(self.label_27)
        self.spinBox = QSpinBox(self.splitter_7)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setStyleSheet(u"background-color:#32576a;")
        self.splitter_7.addWidget(self.spinBox)

        self.verticalLayout_31.addWidget(self.splitter_7)

        self.splitter_8 = QSplitter(self.frame_16)
        self.splitter_8.setObjectName(u"splitter_8")
        self.splitter_8.setOrientation(Qt.Horizontal)
        self.label_28 = QLabel(self.splitter_8)
        self.label_28.setObjectName(u"label_28")
        self.splitter_8.addWidget(self.label_28)
        self.comboBox_5 = QComboBox(self.splitter_8)
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.setObjectName(u"comboBox_5")
        self.comboBox_5.setStyleSheet(u"background-color:#32576a;")
        self.splitter_8.addWidget(self.comboBox_5)

        self.verticalLayout_31.addWidget(self.splitter_8)

        self.splitter_9 = QSplitter(self.frame_16)
        self.splitter_9.setObjectName(u"splitter_9")
        self.splitter_9.setOrientation(Qt.Horizontal)
        self.label_29 = QLabel(self.splitter_9)
        self.label_29.setObjectName(u"label_29")
        self.splitter_9.addWidget(self.label_29)
        self.comboBox_6 = QComboBox(self.splitter_9)
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.setObjectName(u"comboBox_6")
        self.comboBox_6.setStyleSheet(u"background-color:#32576a;")
        self.splitter_9.addWidget(self.comboBox_6)

        self.verticalLayout_31.addWidget(self.splitter_9)


        self.horizontalLayout_16.addWidget(self.frame_16)

        self.splitter_10 = QSplitter(self.frame_12)
        self.splitter_10.setObjectName(u"splitter_10")
        self.splitter_10.setGeometry(QRect(20, 250, 261, 16))
        self.splitter_10.setOrientation(Qt.Horizontal)
        self.label_30 = QLabel(self.splitter_10)
        self.label_30.setObjectName(u"label_30")
        self.splitter_10.addWidget(self.label_30)
        self.comboBox_7 = QComboBox(self.splitter_10)
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.setObjectName(u"comboBox_7")
        self.comboBox_7.setStyleSheet(u"background-color:#32576a;")
        self.splitter_10.addWidget(self.comboBox_7)

        self.verticalLayout_29.addWidget(self.frame_12)

        self.frame_14 = QFrame(self.pagePlcSubContainer)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setFrameShape(QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Raised)

        self.verticalLayout_29.addWidget(self.frame_14)


        self.verticalLayout_26.addWidget(self.pagePlcSubContainer)

        self.mainPages.addWidget(self.pagePLC)

        self.verticalLayout_15.addWidget(self.mainPages)


        self.horizontalLayout_8.addWidget(self.mainContentsContainer)

        self.rightMenuContainer = QCustomSlideMenu(self.mainBodyContent)
        self.rightMenuContainer.setObjectName(u"rightMenuContainer")
        self.rightMenuContainer.setMinimumSize(QSize(200, 0))
        self.rightMenuContainer.setMaximumSize(QSize(200, 615))
        self.verticalLayout_11 = QVBoxLayout(self.rightMenuContainer)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.rightMenuSubContainer = QWidget(self.rightMenuContainer)
        self.rightMenuSubContainer.setObjectName(u"rightMenuSubContainer")
        self.rightMenuSubContainer.setMinimumSize(QSize(200, 0))
        self.verticalLayout_12 = QVBoxLayout(self.rightMenuSubContainer)
        self.verticalLayout_12.setSpacing(5)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(5, 5, 5, 5)
        self.frame_8 = QFrame(self.rightMenuSubContainer)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_8)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_7 = QLabel(self.frame_8)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.label_7)

        self.closeRightmenuButton = QPushButton(self.frame_8)
        self.closeRightmenuButton.setObjectName(u"closeRightmenuButton")
        self.closeRightmenuButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon30 = QIcon()
        icon30.addFile(u":/icons/feather/x-octagon.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.closeRightmenuButton.setIcon(icon30)
        self.closeRightmenuButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_9.addWidget(self.closeRightmenuButton, 0, Qt.AlignRight)


        self.verticalLayout_12.addWidget(self.frame_8)

        self.rightMenuPages = QCustomStackedWidget(self.rightMenuSubContainer)
        self.rightMenuPages.setObjectName(u"rightMenuPages")
        self.pageImagesList = QWidget()
        self.pageImagesList.setObjectName(u"pageImagesList")
        self.verticalLayout_13 = QVBoxLayout(self.pageImagesList)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.label_8 = QLabel(self.pageImagesList)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_13.addWidget(self.label_8)

        self.rightMenuPages.addWidget(self.pageImagesList)
        self.pageMore = QWidget()
        self.pageMore.setObjectName(u"pageMore")
        self.verticalLayout_14 = QVBoxLayout(self.pageMore)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.label_9 = QLabel(self.pageMore)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font)
        self.label_9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_14.addWidget(self.label_9)

        self.rightMenuPages.addWidget(self.pageMore)

        self.verticalLayout_12.addWidget(self.rightMenuPages)


        self.verticalLayout_11.addWidget(self.rightMenuSubContainer)


        self.horizontalLayout_8.addWidget(self.rightMenuContainer, 0, Qt.AlignRight)


        self.verticalLayout_10.addWidget(self.mainBodyContent)

        self.popupNotificationContainer = QCustomSlideMenu(self.mainBodyContainer)
        self.popupNotificationContainer.setObjectName(u"popupNotificationContainer")
        self.verticalLayout_19 = QVBoxLayout(self.popupNotificationContainer)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.popupNotificationSubContainer = QWidget(self.popupNotificationContainer)
        self.popupNotificationSubContainer.setObjectName(u"popupNotificationSubContainer")
        self.verticalLayout_20 = QVBoxLayout(self.popupNotificationSubContainer)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.label_14 = QLabel(self.popupNotificationSubContainer)
        self.label_14.setObjectName(u"label_14")
        font3 = QFont()
        font3.setPointSize(11)
        font3.setBold(True)
        font3.setWeight(75)
        self.label_14.setFont(font3)

        self.verticalLayout_20.addWidget(self.label_14)

        self.frame_9 = QFrame(self.popupNotificationSubContainer)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_13 = QLabel(self.frame_9)
        self.label_13.setObjectName(u"label_13")
        sizePolicy1.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy1)
        self.label_13.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.label_13)

        self.closeNotificationsButton = QPushButton(self.frame_9)
        self.closeNotificationsButton.setObjectName(u"closeNotificationsButton")
        self.closeNotificationsButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.closeNotificationsButton.setIcon(icon30)
        self.closeNotificationsButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_10.addWidget(self.closeNotificationsButton, 0, Qt.AlignRight)


        self.verticalLayout_20.addWidget(self.frame_9)


        self.verticalLayout_19.addWidget(self.popupNotificationSubContainer)


        self.verticalLayout_10.addWidget(self.popupNotificationContainer)

        self.footerContainer = QWidget(self.mainBodyContainer)
        self.footerContainer.setObjectName(u"footerContainer")
        self.horizontalLayout_11 = QHBoxLayout(self.footerContainer)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.frame_10 = QFrame(self.footerContainer)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_15 = QLabel(self.frame_10)
        self.label_15.setObjectName(u"label_15")

        self.horizontalLayout_12.addWidget(self.label_15)


        self.horizontalLayout_11.addWidget(self.frame_10)

        self.sizeGrip = QFrame(self.footerContainer)
        self.sizeGrip.setObjectName(u"sizeGrip")
        self.sizeGrip.setMinimumSize(QSize(30, 30))
        self.sizeGrip.setMaximumSize(QSize(10, 10))
        self.sizeGrip.setFrameShape(QFrame.StyledPanel)
        self.sizeGrip.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_11.addWidget(self.sizeGrip)


        self.verticalLayout_10.addWidget(self.footerContainer)


        self.horizontalLayout.addWidget(self.mainBodyContainer)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.centerMenuPages.setCurrentIndex(1)
        self.mainPages.setCurrentIndex(7)
        self.rightMenuPages.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.menuButton.setToolTip(QCoreApplication.translate("MainWindow", u"Menu", None))
#endif // QT_CONFIG(tooltip)
        self.menuButton.setText("")
#if QT_CONFIG(tooltip)
        self.homebutton.setToolTip(QCoreApplication.translate("MainWindow", u"Home", None))
#endif // QT_CONFIG(tooltip)
        self.homebutton.setText(QCoreApplication.translate("MainWindow", u"Home", None))
#if QT_CONFIG(tooltip)
        self.dataAnalysisbutton.setToolTip(QCoreApplication.translate("MainWindow", u"Data Analysis", None))
#endif // QT_CONFIG(tooltip)
        self.dataAnalysisbutton.setText(QCoreApplication.translate("MainWindow", u"Data Analysis", None))
#if QT_CONFIG(tooltip)
        self.cpu_gpubutton.setToolTip(QCoreApplication.translate("MainWindow", u"CPU/GPU settings", None))
#endif // QT_CONFIG(tooltip)
        self.cpu_gpubutton.setText(QCoreApplication.translate("MainWindow", u"CPU/GPU", None))
#if QT_CONFIG(tooltip)
        self.ObjectDetectionButton.setToolTip(QCoreApplication.translate("MainWindow", u"Object Detection", None))
#endif // QT_CONFIG(tooltip)
        self.ObjectDetectionButton.setText(QCoreApplication.translate("MainWindow", u"Object Detection", None))
#if QT_CONFIG(tooltip)
        self.classificationButton.setToolTip(QCoreApplication.translate("MainWindow", u"Classification", None))
#endif // QT_CONFIG(tooltip)
        self.classificationButton.setText(QCoreApplication.translate("MainWindow", u"Classification", None))
#if QT_CONFIG(tooltip)
        self.QrCodeButton.setToolTip(QCoreApplication.translate("MainWindow", u"OCR BarCode Reader", None))
#endif // QT_CONFIG(tooltip)
        self.QrCodeButton.setText(QCoreApplication.translate("MainWindow", u"OCR Barcode Reader ", None))
#if QT_CONFIG(tooltip)
        self.codeScanGunButton.setToolTip(QCoreApplication.translate("MainWindow", u"Code Scan Gun", None))
#endif // QT_CONFIG(tooltip)
        self.codeScanGunButton.setText(QCoreApplication.translate("MainWindow", u"Code Scan Gun", None))
#if QT_CONFIG(tooltip)
        self.plcButton.setToolTip(QCoreApplication.translate("MainWindow", u"PLC", None))
#endif // QT_CONFIG(tooltip)
        self.plcButton.setText(QCoreApplication.translate("MainWindow", u"PLC", None))
#if QT_CONFIG(tooltip)
        self.cameraSetupbutton.setToolTip(QCoreApplication.translate("MainWindow", u"Camera Setings", None))
#endif // QT_CONFIG(tooltip)
        self.cameraSetupbutton.setText(QCoreApplication.translate("MainWindow", u"Camera Setup", None))
#if QT_CONFIG(tooltip)
        self.settingsButton.setToolTip(QCoreApplication.translate("MainWindow", u"Go-to Settings", None))
#endif // QT_CONFIG(tooltip)
        self.settingsButton.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
#if QT_CONFIG(tooltip)
        self.helpButton.setToolTip(QCoreApplication.translate("MainWindow", u"More Help", None))
#endif // QT_CONFIG(tooltip)
        self.helpButton.setText(QCoreApplication.translate("MainWindow", u"Help", None))
#if QT_CONFIG(tooltip)
        self.logsButton.setToolTip(QCoreApplication.translate("MainWindow", u"Logs", None))
#endif // QT_CONFIG(tooltip)
        self.logsButton.setText(QCoreApplication.translate("MainWindow", u"Logs", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"More Menu", None))
#if QT_CONFIG(tooltip)
        self.closeCenterMenuButton.setToolTip(QCoreApplication.translate("MainWindow", u"Close Menu", None))
#endif // QT_CONFIG(tooltip)
        self.closeCenterMenuButton.setText("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Camera Config", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Logs", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Sysytem Settings", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.label_5.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"RunMo Vision SDK", None))
#if QT_CONFIG(tooltip)
        self.zoomInButton.setToolTip(QCoreApplication.translate("MainWindow", u"Zoom-in", None))
#endif // QT_CONFIG(tooltip)
        self.zoomInButton.setText("")
#if QT_CONFIG(tooltip)
        self.zoomOutButton.setToolTip(QCoreApplication.translate("MainWindow", u"Zoom-out", None))
#endif // QT_CONFIG(tooltip)
        self.zoomOutButton.setText("")
        self.debugMessagesButton.setText("")
#if QT_CONFIG(tooltip)
        self.moreMenuButton.setToolTip(QCoreApplication.translate("MainWindow", u"More", None))
#endif // QT_CONFIG(tooltip)
        self.moreMenuButton.setText("")
#if QT_CONFIG(tooltip)
        self.imagesListButton.setToolTip(QCoreApplication.translate("MainWindow", u"Images List", None))
#endif // QT_CONFIG(tooltip)
        self.imagesListButton.setText("")
#if QT_CONFIG(tooltip)
        self.minimizeButton.setToolTip(QCoreApplication.translate("MainWindow", u"Minimize Window", None))
#endif // QT_CONFIG(tooltip)
        self.minimizeButton.setText("")
#if QT_CONFIG(tooltip)
        self.restoreButton.setToolTip(QCoreApplication.translate("MainWindow", u"Restore Window", None))
#endif // QT_CONFIG(tooltip)
        self.restoreButton.setText("")
#if QT_CONFIG(tooltip)
        self.closeButton.setToolTip(QCoreApplication.translate("MainWindow", u"Close Window", None))
#endif // QT_CONFIG(tooltip)
        self.closeButton.setText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Home", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Data Analysis", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"CPU/GPU Performace Metrics", None))
#if QT_CONFIG(tooltip)
        self.openCameraButton.setToolTip(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#endif // QT_CONFIG(tooltip)
        self.openCameraButton.setText(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#if QT_CONFIG(tooltip)
        self.labelImagesButton.setToolTip(QCoreApplication.translate("MainWindow", u"Label Image", None))
#endif // QT_CONFIG(tooltip)
        self.labelImagesButton.setText(QCoreApplication.translate("MainWindow", u"label Images", None))
#if QT_CONFIG(tooltip)
        self.captureImageButton.setToolTip(QCoreApplication.translate("MainWindow", u"Capture Image", None))
#endif // QT_CONFIG(tooltip)
        self.captureImageButton.setText(QCoreApplication.translate("MainWindow", u"Capture Image", None))
#if QT_CONFIG(tooltip)
        self.trainButton.setToolTip(QCoreApplication.translate("MainWindow", u"Train Model", None))
#endif // QT_CONFIG(tooltip)
        self.trainButton.setText(QCoreApplication.translate("MainWindow", u"Train", None))
#if QT_CONFIG(tooltip)
        self.detectRealTimeButton.setToolTip(QCoreApplication.translate("MainWindow", u"Detect Using Camera", None))
#endif // QT_CONFIG(tooltip)
        self.detectRealTimeButton.setText(QCoreApplication.translate("MainWindow", u"Detect In Real-Time", None))
#if QT_CONFIG(tooltip)
        self.detectImageButton.setToolTip(QCoreApplication.translate("MainWindow", u"Detect From Image", None))
#endif // QT_CONFIG(tooltip)
        self.detectImageButton.setText(QCoreApplication.translate("MainWindow", u"Detect From Image", None))
#if QT_CONFIG(tooltip)
        self.downloadModelButton.setToolTip(QCoreApplication.translate("MainWindow", u"Download Model", None))
#endif // QT_CONFIG(tooltip)
        self.downloadModelButton.setText(QCoreApplication.translate("MainWindow", u"Download Model", None))
#if QT_CONFIG(tooltip)
        self.gentfrecordsButton.setToolTip(QCoreApplication.translate("MainWindow", u"Generate TF records", None))
#endif // QT_CONFIG(tooltip)
        self.gentfrecordsButton.setText(QCoreApplication.translate("MainWindow", u"TF records", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Object Detection", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Image 1", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Image 2 ", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Results Label", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Training Functions", None))
#if QT_CONFIG(tooltip)
        self.openCamerabtnclassifier.setToolTip(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#endif // QT_CONFIG(tooltip)
        self.openCamerabtnclassifier.setText(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#if QT_CONFIG(tooltip)
        self.captureBtnClassifier.setToolTip(QCoreApplication.translate("MainWindow", u"Capture Image", None))
#endif // QT_CONFIG(tooltip)
        self.captureBtnClassifier.setText(QCoreApplication.translate("MainWindow", u"Capture", None))
#if QT_CONFIG(tooltip)
        self.plotGraphbuttonClassifier.setToolTip(QCoreApplication.translate("MainWindow", u"Plot Graph", None))
#endif // QT_CONFIG(tooltip)
        self.plotGraphbuttonClassifier.setText(QCoreApplication.translate("MainWindow", u"Plot Performace Metrics", None))
#if QT_CONFIG(tooltip)
        self.finalClassiybutton.setToolTip(QCoreApplication.translate("MainWindow", u"Classify", None))
#endif // QT_CONFIG(tooltip)
        self.finalClassiybutton.setText(QCoreApplication.translate("MainWindow", u"Classify", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"OCR Barcode Reader", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Code Scan Gun", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"PLC Type", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Modbus1", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Modbus2", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Modbus3", None))

        self.plc1Button.setText(QCoreApplication.translate("MainWindow", u"PLC 1", None))
        self.plc2Button.setText(QCoreApplication.translate("MainWindow", u"PLC 2", None))
        self.plc3Button.setText(QCoreApplication.translate("MainWindow", u"PLC 3", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.comboBox_2.setItemText(0, QCoreApplication.translate("MainWindow", u"Network Connection(TCP)", None))

        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Serial Port", None))
        self.comboBox_3.setItemText(0, QCoreApplication.translate("MainWindow", u"COM 1", None))
        self.comboBox_3.setItemText(1, QCoreApplication.translate("MainWindow", u"COM 2", None))
        self.comboBox_3.setItemText(2, QCoreApplication.translate("MainWindow", u"COM 3", None))

        self.label_26.setText(QCoreApplication.translate("MainWindow", u"PLC IP ", None))
        self.comboBox_4.setItemText(0, QCoreApplication.translate("MainWindow", u"192.168.1.15", None))
        self.comboBox_4.setItemText(1, QCoreApplication.translate("MainWindow", u"192.168.2.17", None))

        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Feedback timeout", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Baud Rate", None))
        self.comboBox_5.setItemText(0, QCoreApplication.translate("MainWindow", u"9600", None))
        self.comboBox_5.setItemText(1, QCoreApplication.translate("MainWindow", u"9500", None))
        self.comboBox_5.setItemText(2, QCoreApplication.translate("MainWindow", u"9400", None))
        self.comboBox_5.setItemText(3, QCoreApplication.translate("MainWindow", u"New Item", None))

        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Interface", None))
        self.comboBox_6.setItemText(0, QCoreApplication.translate("MainWindow", u"560", None))
        self.comboBox_6.setItemText(1, QCoreApplication.translate("MainWindow", u"460", None))
        self.comboBox_6.setItemText(2, QCoreApplication.translate("MainWindow", u"360", None))
        self.comboBox_6.setItemText(3, QCoreApplication.translate("MainWindow", u"260", None))

        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Number of Iterations", None))
        self.comboBox_7.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.comboBox_7.setItemText(1, QCoreApplication.translate("MainWindow", u"2", None))
        self.comboBox_7.setItemText(2, QCoreApplication.translate("MainWindow", u"3", None))
        self.comboBox_7.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))
        self.comboBox_7.setItemText(4, QCoreApplication.translate("MainWindow", u"5", None))

        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Right Menu", None))
#if QT_CONFIG(tooltip)
        self.closeRightmenuButton.setToolTip(QCoreApplication.translate("MainWindow", u"Close Menu", None))
#endif // QT_CONFIG(tooltip)
        self.closeRightmenuButton.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Images", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"More...", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Debug!", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Error Messages", None))
#if QT_CONFIG(tooltip)
        self.closeNotificationsButton.setToolTip(QCoreApplication.translate("MainWindow", u"Close Notification", None))
#endif // QT_CONFIG(tooltip)
        self.closeNotificationsButton.setText("")
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Copyright Runmo Intelligent Research Department", None))
    # retranslateUi


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())