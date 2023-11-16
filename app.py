#Import necessary libraries
from flask import Flask, render_template, request, flash
 
import numpy as np
import os
 
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
 
#-----------------------------Deep Learning Model Configuration----------------------------
model_brain =load_model("models/braindisease.h5")
model_skin=load_model("models/skindisease.h5")
#------------------------------------------------------------------------------------------
#------------------------------------------Skin Disease-------------------------------------
def predict_skin_disease(skinImage):
  print(skinImage)
  test_image = load_img(skinImage, target_size = (224, 224)) # load image with custom sizing

   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model_skin.predict(test_image).round(3) # predict diseased palnt or not
 
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Ekzama Skin" # if index is 0 then Ekzama
  elif pred == 1:
      return 'Acne Skin' # if index is 1 then Acne 
  elif pred == 2:
      return 'Malign Skin'  # if index is  2 then Malign
  elif pred == 3:
      return 'Normal Skin'# if index is 3 then Normal
  else:
    print("No Image Found!")
#----------------------------------------------------------------------------------------------- 
#-----------------------------------Brain disease-----------------------------------------------
def predict_brain_disease(brainImage):
  print(brainImage)
  test_image = load_img(brainImage, target_size = (224, 224)) # load image 

   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model_skin.predict(test_image).round(3) # predict diseased palnt or not
 
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Alzeimer Mild Demented"# if index is 0 then Alzeimer mild
  elif pred == 1:
      return 'Alzeimer Moderate Demented'# if index is 1 then Alzeimer mild
  elif pred == 2:
      return 'Glioma'# if index is 2 then Glioma
  elif pred == 3:
      return 'Meningioma'# if index is 3 thenmeningioma
  elif pred == 4:
      return 'No-Tumor'# if index is 4 then No tumor
  else:
    print("No Image Found!")
     
 
# Create flask instance
app = Flask(__name__)
app.secret_key = "secret key"
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
@app.route("/info", methods=['GET'])
def information():
    return render_template('information.html')
@app.route("/info-brain", methods=['GET'])
def informationbrain():
    return render_template('informationbrain.html')
@app.route("/help", methods=['GET'])
def help():
    return render_template('help.html')
@app.route("/result/<disease>",methods=['GET'])
def result():
    return render_template('result.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict-brain", methods = ['GET','POST'])
def predictbrain():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename 
        if filename=="":
            flash("Please Insert Image", 'error')  
            return render_template('index.html')
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred = predict_brain_disease(brainImage=file_path)
               
        return render_template('result.html', pred_output = pred, user_image = file_path)
@app.route("/predict-skin", methods = ['GET','POST'])
def predictskin():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename 
        if filename=="":
            flash("Please Insert Image", 'error')  
            return render_template('index.html')
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred= predict_skin_disease(skinImage=file_path)
               
        return render_template('result.html', pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    #app.run(debug = False)
    app.run(threaded=True) 
