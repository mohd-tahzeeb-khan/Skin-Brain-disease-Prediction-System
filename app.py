#Import necessary libraries
from flask import Flask, render_template, request, flash
 
import numpy as np
import os
 
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
#-----------------------------HTML Pages Configuration------------------------------------
result_page='result.html'
#-----------------------------------------------------------------------------------------
#-----------------------------Deep Learning Model Configuration----------------------------
model_brain =load_model("models/brain_disease ETC retrain copy 10 epochsFinal.h5")
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
    data={
        "disease_name":"Eczema Skin Disease",
        "abstraction":"Atopic dermatitis(Eczema) is a condition that causes dry, itchy and inflamed skin. It's common in young children but can occur at any age. Atopic dermatitis is long lasting(Chronic) and tends to flare sometimes. It can be irritating but it's not contagious. <br> People with atopic dermatites are at risk of diveloping food allergies, high fever and asthma. <br> Moisturizing regularly and following other skin care habits can relieve itching and prevent new outbreaks(flares). Treatment may also include medicated ointments or cream.",
        "symptom_define":"In infants, the itchy rash can lead to an oozing, crusting condition, mainly on the face and scalp. It can also happen on their arms, legs, back, and chest. Newborn babies can show symptoms within the first few weeks or months after birth. The rash usually happens on your face, the backs of your knees, wrists, hands, or feet. Your skin will probably be very dry, thick, or scaly. In fair-skinned people, these areas may start out reddish and then turn brown. In darker-skinned people, eczema can affect skin pigments, making the affected area lighter or darker. Developing a basic skin care routine may help prevent eczema flares. The following tips may help reduce the drying effects of bathing:",
        "symptom_list":"<li>Dry, cracked skin</li><li>Itchiness (pruritus)</li><li>Rash on swollen skin that varies in color depending on your skin color</li><li>Oozing and crusting, Thickened skin</li><li>Small, raised bumps, on brown or Black skin</li><li>Darkening of the skin around the eyes Raw, sensitive skin from scratching</li>",
        "treatment":"<li>Moisturize your skin at least twice a day</li><li>Take a daily bath or Shower</li><li>Use a gentle, nonsoap cleaner</li>"
    }
    return data # if index is 0 then Ekzama
  elif pred == 1:
        data={
        "disease_name":"Acne & Pores Skin",
        "abstraction":"Acne vulgaris, commonly known as acne, is a chronic inflammatory skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age. The exact cause of acne is not fully understood, but it is thought to be influenced by a combination of genetic, hormonal, and environmental factors. Androgens, hormones that are produced in increased amounts during puberty, can stimulate sebum production and contribute to the development of acne. Other factors that may play a role include bacterial colonization of hair follicles, inflammation, and stress. Acne can be a source of significant emotional distress, particularly for adolescents and young adults. It can lead to feelings of self-consciousness, anxiety, and depression. In severe cases, acne can also cause scarring. There are a variety of treatment options available for acne, depending on the severity of the condition. Topical treatments, such as benzoyl peroxide, retinoids, and azelaic acid, are often used as first-line therapy. Oral antibiotics may be prescribed for moderate to severe acne. In severe cases, isotretinoin, a powerful oral medication, may be used.",
        "symptom_define":"Acne, also known as acne vulgaris, is a common skin condition that occurs when hair follicles become plugged with dead skin cells and oil (sebum). This can lead to the formation of various lesions, including blackheads, whiteheads, papules, pustules, nodules, and cysts. Acne typically affects the face, back, chest, and shoulders, and it is most common during adolescence, although it can occur at any age. <br> The exact cause of acne is not fully understood, but it is thought to be influenced by a combination of genetic, hormonal, and environmental factors. Here is a breakdown of the main factors that contribute to acne development:",
        "symptom_list":"<li>Excessive Sebum Production: Sebum is an oily substance produced by sebaceous glands attached to hair follicles. During puberty, hormonal changes can lead to increased sebum production, making the skin more oily and prone to clogging.</li><li>Clogged Hair Follicles: When dead skin cells mix with excess sebum, they can clog the openings of hair follicles, leading to the formation of comedones, commonly known as blackheads and whiteheads.</li><li>Bacterial Overgrowth: The bacteria Propionibacterium acnes (P. acnes) naturally resides on the skin. However, when hair follicles become clogged, P. acnes can multiply and trigger inflammation, leading to the development of papules, pustules, nodules, and cysts.</li><li>Inflammation: Inflammation is a key factor in acne development. When P. acnes bacteria interact with sebum and dead skin cells, they release inflammatory substances that trigger an immune response. This inflammation leads to the redness, swelling, and tenderness associated with acne lesions.</li><li>Genetics: Acne tends to run in families, suggesting a genetic predisposition to the condition. Genetic factors can influence sebum production, hair follicle structure, and susceptibility to bacterial overgrowth.</li><li>Hormonal Changes: Hormonal fluctuations, particularly during adolescence and puberty, can play a significant role in acne development. Androgen hormones, which are produced in higher amounts during puberty, stimulate sebum production and contribute to acne formation.",
        "treatment":"Acne and pores are closely related, as acne is caused by a blockage of pores. Pores are small openings in the skin that allow sebum, an oily substance produced by sebaceous glands, to reach the surface of the skin. Sebum helps to keep the skin lubricated and protected. However, when too much sebum is produced or when dead skin cells accumulate in the pores, they can become clogged. This blockage can then lead to the formation of acne lesions.<li>Wash your face twice a day: Washing your face twice a day with a gentle cleanser can help to remove excess oil, dead skin cells, and bacteria from the pores.</li><li>Use oil-free skincare products: Oil-free skincare products can help to prevent clogged pores.</li><li>Avoid picking or popping pimples: Picking or popping pimples can lead to scarring and infection.</li><li>Maintain a healthy diet: A healthy diet can help to improve overall skin health and may help to reduce the risk of acne.</li><li>Manage stress: Stress can aggravate existing acne or make it more difficult to manage.</li>"
    }
        return data # if index is 1 then Acne 
  elif pred == 2:
      data={
        "disease_name":"Malign Skin Disease",
        "abstraction":"Melanoma, also redundantly known as malignant melanoma,is a type of cancer that develops from the pigment-producing cells known as melanocytes. Melanomas typically occur in the skin, but may rarely occur in the mouth, intestines, or eye (uveal melanoma). In women, they most commonly occur on the legs, while in men, they most commonly occur on the back. About 25% of melanomas develop from moles. Changes in a mole that can indicate melanoma include an increase in size, irregular edges, change in color, itchiness, or skin breakdown. The primary cause of melanoma is ultraviolet light (UV) exposure in those with low levels of the skin pigment melanin. The UV light may be from the sun or other sources, such as tanning devices. Those with many moles, a history of affected family members, and poor immune function are at greater risk. A number of rare genetic conditions, such as xeroderma pigmentosum, also increase the risk. Diagnosis is by biopsy and analysis of any skin lesion that has signs of being potentially cancerous.",
        "symptom_define":"Early signs of melanoma are changes to the shape or color of existing moles or, in the case of nodular melanoma, the appearance of a new lump anywhere on the skin. At later stages, the mole may itch, ulcerate, or bleed. Early signs of melanoma are summarized by the mnemoni",
        "symptom_list":"<li>Asymmetry</li><li>Borders (irregular with edges and corners)</li><li>Colour (variegated)</li><li>Diameter (greater than 6 mm (0.24 in), about the size of a pencil eraser)</li><li>Evolving over time</li><li>Elevated above the skin surface</li><li>Firm to the touch</li><li>Growing</li>",
        "treatment":"Looking at or visually inspecting the area in question is the most common method of suspecting a melanoma. Moles that are irregular in color or shape are typically treated as candidates. To detect melanomas (and increase survival rates), it is recommended to learn to recognize them (see 'ABCDE' mnemonic), to regularly examine moles for changes (shape, size, color, itching or bleeding) and to consult a qualified physician when a candidate appears. In-person inspection of suspicious skin lesions is more accurate than visual inspection of images of suspicious skin lesions. When used by trained specialists, dermoscopy is more helpful to identify malignant lesions than use of the naked eye alone. Reflectance confocal microscopy may have better sensitivity and specificity than dermoscopy in diagnosing cutaneous melanoma but more studies are needed to confirm this result. However, many melanomas present as lesions smaller than 6 mm in diameter, and all melanomas are malignant when they first appear as a small dot. Physicians typically examine all moles, including those less than 6 mm in diameter. Seborrheic keratosis may meet some or all of the ABCD criteria, and can lead to false alarms. Doctors can generally distinguish seborrheic keratosis from melanoma upon examination or with dermatoscopy. Some advocate replacing 'enlarging' with 'evolving': moles that change and evolve are a concern. Alternatively, some practitioners prefer 'elevation'. Elevation can help identify a melanoma, but lack of elevation does not mean that the lesion is not a melanoma. Most melanomas in the US are detected before they become elevated. By the time elevation is visible, they may have progressed to the more dangerous invasive stage."
    }
      
      return data  # if index is  2 then Malign
  elif pred == 3:
      data={
        "disease_name":"Good Skin.",
        "abstraction":"Good skin is typically characterized by its smooth texture, even tone, and healthy glow. It is free of blemishes, dryness, and excessive oiliness, and it reflects an overall sense of vitality and well-being. Here's a more detailed description of the qualities that define good skin. Remember, good skin is not just about genetics or luck; it is also a result of conscious choices, consistent care, and a commitment to overall health and well-being. By adopting healthy habits, incorporating a personalized skincare routine, and addressing any underlying skin concerns, you can nurture your skin's natural beauty and achieve a healthy, glowing complexion.",
        "symptom_define":"---",
        "symptom_list":"---",
        "treatment":"<li>Choose gentle skincare products: Avoid harsh soaps and cleansers that can strip away natural oils and irritate the skin.</li><li>Moisturize regularly: Moisturizing helps maintain skin hydration and prevent dryness.</li><li>Exfoliate regularly: Exfoliating removes dead skin cells and promotes skin cell turnover.</li><li>Protect your skin from the sun: Regular use of sunscreen with an SPF of 30 or higher protects against sun damage and premature aging.</li><li>Manage stress effectively: Chronic stress can worsen skin conditions and accelerate aging.</li><li>Avoid smoking: Smoking damages skin cells and accelerates aging.</li><li>Maintain a healthy sleep routine: Adequate sleep allows the skin to repair and regenerate.</li>"
    }
      return data# if index is 3 then Normal
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
    data={
        "disease_name":"Alzeimer Mild Demented",
        "abstraction":"Alzheimer's disease is a progressive neurodegenerative disease that causes memory loss and cognitive decline. Mild dementia is an early stage of Alzheimer's disease characterized by subtle memory problems and other cognitive changes. People with mild dementia may still be able to live independently and perform most of their daily activities, but they may need some assistance with complex tasks. There is no cure for Alzheimer's disease, but there are treatments that can help slow the progression of the disease and manage symptoms. Early diagnosis and treatment are important for people with mild dementia, as this can help them maintain their independence and quality of life for as long as possible.",
        "symptom_define":"Alzheimer's disease is a progressive neurodegenerative disease that causes memory loss and cognitive decline. Mild dementia is an early stage of Alzheimer's disease characterized by subtle memory problems and other cognitive changes. People with mild dementia may still be able to live independently and perform most of their daily activities, but they may need some assistance with complex tasks.",
        "symptom_list":"<li>Memory loss, especially for recent events</li><li>Difficulty finding words</li><li>Getting lost or disoriented</li><li>Changes in mood or personality</li><li>Loss of interest in activities that were once enjoyed</li><li>Changes in sleep patterns</li>",
        "treatment":"There is no single test that can diagnose Alzheimer's disease, but doctors can use a combination of tests to make a diagnosis. These tests may include: A medical history and physical exam, Cognitive tests to assess memory, thinking skills, and other mental abilities, Brain scans, such as an MRI or CT scan, to look for changes in the brain, Blood tests to rule out other possible causes of dementia<br>There is no cure for Alzheimer's disease, but there are treatments that can help slow the progression of the disease and manage symptoms. These treatments may include:<li>Medications that can help with memory and other cognitive problems</li><li>Counseling and support for people with dementia and their caregivers</li><li>Changes in lifestyle, such as eating a healthy diet and getting regular exercise</li>"
    }
    return data # if  index is 0 then Alzeimer mild
  elif pred == 1:
      data={
        "disease_name":"Alzeimer Modarate Demented",
        "abstraction":"Alzheimer's disease is a progressive neurodegenerative disease that causes memory loss and cognitive decline. Mild dementia is an early stage of Alzheimer's disease characterized by subtle memory problems and other cognitive changes. People with mild dementia may still be able to live independently and perform most of their daily activities, but they may need some assistance with complex tasks. There is no cure for Alzheimer's disease, but there are treatments that can help slow the progression of the disease and manage symptoms. Early diagnosis and treatment are important for people with mild dementia, as this can help them maintain their independence and quality of life for as long as possible.",
        "symptom_define":"Alzheimer's disease is a progressive neurodegenerative disease that causes memory loss and cognitive decline. Mild dementia is an early stage of Alzheimer's disease characterized by subtle memory problems and other cognitive changes. People with mild dementia may still be able to live independently and perform most of their daily activities, but they may need some assistance with complex tasks.",
        "symptom_list":"<li>Memory loss, especially for recent events</li><li>Difficulty finding words</li><li>Getting lost or disoriented</li><li>Changes in mood or personality</li><li>Loss of interest in activities that were once enjoyed</li><li>Changes in sleep patterns</li>",
        "treatment":"There is no single test that can diagnose Alzheimer's disease, but doctors can use a combination of tests to make a diagnosis. These tests may include: A medical history and physical exam, Cognitive tests to assess memory, thinking skills, and other mental abilities, Brain scans, such as an MRI or CT scan, to look for changes in the brain, Blood tests to rule out other possible causes of dementia<br>There is no cure for Alzheimer's disease, but there are treatments that can help slow the progression of the disease and manage symptoms. These treatments may include:<li>Medications that can help with memory and other cognitive problems</li><li>Counseling and support for people with dementia and their caregivers</li><li>Changes in lifestyle, such as eating a healthy diet and getting regular exercise</li>"
    }
      return data # if index is 1 then Alzeimer mild
  elif pred == 2:
      data={
        "disease_name":"Glioma Brain Disease",
        "abstraction":"Glioma is a type of tumor that originates from glial cells, which are supportive cells in the central nervous system (CNS). Gliomas are the most common type of primary brain tumor, accounting for about 80% of all cases. They can occur in any part of the CNS, including the brain, spinal cord, and nerve roots.<br>Diagnosis of glioma is typically made using a combination of imaging tests, such as MRI or CT scans, and biopsies, which involve removing a small sample of tissue for examination under a microscope.<br>The prognosis for glioma depends on the grade of the tumor, the patient's age and overall health, and the response to treatment. Low-grade gliomas are generally slow-growing and can be managed with surgery, radiation therapy, and chemotherapy. High-grade gliomas are more aggressive and have a poorer prognosis. However, advances in treatment have improved survival rates for patients with high-grade gliomas in recent years.<br>If you are concerned that you or someone you know may have glioma, it is important to see a doctor for diagnosis and treatment. Early diagnosis and treatment can improve the chances of a good outcome.",
        "symptom_define":"The symptoms of glioma can vary depending on the location and size of the tumor. Some common symptoms include:",
        "symptom_list":"<li>Headaches: These may be constant, intermittent, or worse in the morning.</li><li>Seizures: These are more common with tumors located in the temporal lobe of the brain</li><li>Focal neurological deficits: These can include weakness or paralysis on one side of the body, difficulty with speech or vision, or problems with balance or coordination.</li><li>Cognitive changes: These can include memory problems, difficulty with concentration, or changes in personality or behavior.</li>",
        "treatment":"Treatment options for glioma depend on the grade of the tumor, the patient's age and overall health, and the location of the tumor. Treatment options may include: <li>Surgery: This is the primary treatment for low-grade gliomas. The goal of surgery is to remove as much of the tumor as possible without causing damage to healthy brain tissue.</li><li>Radiation therapy: This uses high-energy beams to kill cancer cells. Radiation therapy may be used after surgery to help reduce the risk of the tumor coming back, or it may be used as the primary treatment for high-grade gliomas.</li><li>Chemotherapy: This uses drugs to kill cancer cells throughout the body. Chemotherapy may be used in conjunction with radiation therapy or on its own for high-grade gliomas.</li><li>Targeted therapy: This uses drugs that specifically target cancer cells. Targeted therapy is a newer treatment option for glioma and is still under investigation.</li>"
    }
      return data # if index is 2 then Glioma
  elif pred == 3:
      data={
        "disease_name":"Meningioma Brain Disease",
        "abstraction":"A meningioma is a type of tumor that arises from the meninges, which are the membranes that surround the brain and spinal cord. These tumors are the most common type of primary brain tumor, accounting for approximately 30% of all cases. They are typically benign (noncancerous) and slow-growing, and they rarely spread to other parts of the body.",
        "symptom_define":"The symptoms of meningioma depend on the location and size of the tumor. Some common symptoms include:",
        "symptom_list":"<li>Headaches</li><li>Seizures</li><li>Vision problems</li><li>Numbness or weakness in the arms or legs</li><li>Difficulty with balance or coordination</li><li>Memory problems</li><li>Changes in personality or behavior</li>",
        "treatment":"Treatment options for meningioma depend on the location and size of the tumor, the patient's age and overall health, and the presence of any symptoms. Treatment options may include: <li>Observation: For small, slow-growing tumors that are not causing any symptoms, observation may be the best course of treatment. The tumor will be monitored with regular imaging tests to check for any changes.</li><li>Surgery: This is the primary treatment for meningiomas that are causing symptoms or that are growing rapidly. The goal of surgery is to remove as much of the tumor as possible without causing damage to healthy brain tissue.</li><li>Radiation therapy: This uses high-energy beams to kill cancer cells. Radiation therapy may be used after surgery to help reduce the risk of the tumor coming back, or it may be used as the primary treatment for meningiomas that are difficult to remove surgically.</li><li>Stereotactic radiosurgery: This is a type of radiation therapy that uses very precisely focused beams to target the tumor. Stereotactic radiosurgery is often used for meningiomas that are located in difficult-to-reach areas of the brain.</li>"
    }
      return data # if index is 3 thenmeningioma
  elif pred == 4:
      data={
        "disease_name":"Healthy Brain",
        "abstraction":"Atopic dermatitis(Eczema) is a condition that causes dry, itchy and inflamed skin. It's common in young children but can occur at any age. Atopic dermatitis is long lasting(Chronic) and tends to flare sometimes. It can be irritating but it's not contagious. <br> People with atopic dermatites are at risk of diveloping food allergies, high fever and asthma. <br> Moisturizing regularly and following other skin care habits can relieve itching and prevent new outbreaks(flares). Treatment may also include medicated ointments or cream.",
        "symptom_define":"In infants, the itchy rash can lead to an oozing, crusting condition, mainly on the face and scalp. It can also happen on their arms, legs, back, and chest. Newborn babies can show symptoms within the first few weeks or months after birth. The rash usually happens on your face, the backs of your knees, wrists, hands, or feet. Your skin will probably be very dry, thick, or scaly. In fair-skinned people, these areas may start out reddish and then turn brown. In darker-skinned people, eczema can affect skin pigments, making the affected area lighter or darker. Developing a basic skin care routine may help prevent eczema flares. The following tips may help reduce the drying effects of bathing:",
        "symptom_list":"<li>Dry, cracked skin</li><li>Itchiness (pruritus)</li><li>Rash on swollen skin that varies in color depending on your skin color</li><li>Oozing and crusting, Thickened skin</li><li>Small, raised bumps, on brown or Black skin</li><li>Darkening of the skin around the eyes Raw, sensitive skin from scratching</li>",
        "treatment":"<li>Moisturize your skin at least twice a day</li><li>Take a daily bath or Shower</li><li>Use a gentle, nonsoap cleaner</li>"
    }
      return data # if index is 4 then No tumor
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
        data = predict_brain_disease(brainImage=file_path)
        return render_template(result_page, data = data, user_image = file_path)
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
        data= predict_skin_disease(skinImage=file_path)
        return render_template(result_page, data = data, user_image = file_path)
# For local system & cloud
if __name__ == "__main__":
    #app.run(debug = False)
    app.run(threaded=True) 
