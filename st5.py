import os
import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import streamlit as st
import pydot
import graphviz


st.set_option('deprecation.showfileUploaderEncoding', False)

ele = """<style>
body {
  background-image: url('arch1.jpg');
}
</style>"""

st.markdown(ele,unsafe_allow_html=True)







import streamlit as st
import pandas as pd

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False






def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def show_home():
    st.subheader("Home")
    st.image('deepfakescrop-13.gif')
    st.subheader("What is Deepfake?")

    st.write("Deepfake technology can create hyper-realistic fake videos from any given face picture, and today it would only take less than 10 minutes to create one. Deepfake videos bear potential risks such as targeted reputation attacks, misinformation, and propaganda.")

    st.subheader("Our Solution")

    st.write("We have developed a deepfake detection technology that can identify when a video is altered to mislead the viewer. In simpler words, we detect deepfakes and allowing everyone to see if a particular video is deepfake or not.")
    st.subheader("Proposed System Architecture")
    st.image("arch1.jpg", width = 800)
    st.write("The Deepfake videos are converted to frames and fed into face detector (MTCNN) to obtain the image feature vectors by using MobileNet. These Feature vectors are given as input to the pretrained MobileNet and then perform Transfer Learning in Pretrained MobileNet Neural Network to classify the Videos as Fake or Real.")
    st.subheader("Results")
    st.image("arch2.jpg" , width = 800)
    st.subheader("Research Paper")
    st.image("paper.jpg" , width = 800)
    st.markdown("""<a href="https://www.researchgate.net/publication/341039783_Deepfake_Video_Forensics_based_on_Transfer_Learning">View Full Paper</a>""", unsafe_allow_html=True)
    st.subheader("About Us")
    st.write("Rahul U")
    st.write("Ragul M")
    st.write("Raja Vignesh K")
    st.subheader("Mentor")
    st.write("Ms. Tejeswinee.K")





def main():
	"""Deepfake Detection"""
  
	st.title("Deepfake Detection App")

	menu = ["Home","Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu)



	if choice == "Home":
		show_home()


        #st.write("Deepfake technology can create hyper-realistic fake videos from any given face picture, and today it would only take less than 10 minutes to create one. Deepfake videos bear potential risks such as targeted reputation attacks, misinformation, and propaganda.")
        #st.subheader("Our Solution?")
        #st.write("We have developed a deepfake detection technology that can identify when a video is altered to mislead the viewer. In simpler words, we detect deepfakes and allowing everyone to see if a particular video is deepfake or not.")
        
    

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("Logged In as {}".format(username))

				process()

				
			else:
				st.warning("Incorrect Username/Password")
        

	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")


def process():
  
  model = load_model('fullset.h5')
  

  
  
  
  uploaded_f = st.file_uploader("Upload a Video...", type="mp4")

  if uploaded_f is not None :
        g = io.BytesIO(uploaded_f.read())  ## BytesIO Object
        temporary_location = "test.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

    # close file
        out.close()
        st.video('test.mp4')
        if st.button("Scan Now"):
            
            detector = dlib.get_frontal_face_detector()
            cap = cv2.VideoCapture("test.mp4")
            np.set_printoptions(suppress=True)
            m=0
            n=0
            count = 0
            frameRate = cap.get(5)
            while cap.isOpened() and count <=12:
                frameId = cap.get(1)
                ret, frame = cap.read()
                if ret != True:
                    break
                if frameId % ((int(frameRate)+1)*1) == 0:
                    face_rects, scores, idx = detector.run(frame, 0)
                    for i, d in enumerate(face_rects):
                        x1 = d.left()
                        y1 = d.top()
                        x2 = d.right()
                        y2 = d.bottom()
                        crop_img = frame[y1:y2, x1:x2]
                        if crop_img.size == 0:
                            continue
                        crop_img = cv2.resize(crop_img, (224, 224))

              #data = data.reshape(-1, 128, 128, 3)
                        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                        image_array = np.asarray(crop_img)
                        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 
                        data[0] = normalized_image_array
                        prediction = model.predict(data)
                        count+=1
                        print(prediction)
                        if prediction[0][0] > 0.50:
                            print("fake")
                            m=m+1
               
                        else:
                            print("real")
                            n=n+1
                
            if m > n :
                m=int((m/(m+n))*100)
                a=int(m)
                x = str(a)
                predict="Fake Video"+"\t" + x + "  %"
                #st.image(crop_img, caption="Deepfake Face Detected")
                st.markdown("**Fake Video**   {} %".format(x))

            else:
                n=int((n/(m+n))*100)
                b=int(n)
                y = str(b)
                predict="Real Video"+ "\t" + y + "  %"
                #st.image(crop_img , caption = "Real Face Detected")
                st.markdown("**Real Video**   {}%".format(y))
        




if __name__ == '__main__':
	main()