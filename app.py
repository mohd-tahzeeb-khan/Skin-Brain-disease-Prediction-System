#Import necessary libraries
from flask import Flask, render_template, request, flash
 
import numpy as np
import os
 
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
 
#load model
model =load_model("model/v5_pred_cott_diis1.h5")
 
print('@@ Model loaded')
 
 
def pred_cot_dieas(cott_plant):
  print(cott_plant)
  test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Bacterial Blight Cotton Plant", 'bacterial_blight_cotton.html' # if index 0 burned leaf
  elif pred == 1:
      return 'Curl Virus', 'curl_Disease.html' # # if index 1
  elif pred == 2:
      return 'Diseased Cotton Leaf', 'disease_plant_leaf.html'  # if index 2  fresh leaf
  elif pred == 3:
      return 'Diseased Cotton Plant', 'disease_plant.html' # # if index 1
  elif pred == 4:
      return 'Fresh Cotton Leaf', 'healthy_plant_leaf.html'  # if index 2  fresh leaf
  elif pred == 5:
      return 'Healthy Cotton Plant', 'healthy_plant.html'  # if index 2  fresh leaf
  elif pred == 6:
      return 'Fussarium Wilt', 'Fussarium_wilt.html'  # if index 2  fresh leaf
  else:
    print("No Image Found!")
 
#------------>>pred_cot_dieas<<--end
     
 
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
@app.route("/help", methods=['GET'])
def help():
    return render_template('help.html')

     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
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
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    #app.run(debug = False)
    app.run(threaded=True) 
