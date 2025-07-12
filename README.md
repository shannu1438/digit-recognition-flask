# ✍ Handwritten Digit Recognition Web App

This is a machine learning web application that recognizes handwritten digits (0 to 9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The app has a user-friendly web interface created using Flask.



##  Features

-  Trained on MNIST dataset  
-  Accepts uploaded digit images  
-  Predicts digit using trained model  
-  Built with Python, Flask, TensorFlow  
-  Fast and accurate predictions  

---
## Teck Stack
-   Python
-   Flask
-   TensFlow / Keras
-   HTML/CSS

##  Project Structure


digit_recognition_project/
├── app.py                # Flask backend
├── digit_model.h5        # Trained CNN model
└── templates/
    └── index.html        # Frontend UI




## How to Run Locally

1. Clone this repository:
   
   git clone https://github.com/shannu1438/digit-recognition-flask.git
   cd digit-recognition-flask
   

2. Install dependencies:
   
   pip install flask tensorflow numpy pillow
   

3. Run the app:
   
   python app.py
   

4. Open my browser and visit:
   
   http://127.0.0.1:5000
   

5. Upload a digit image and get prediction ✅



## Author

*Shaik Shannu*  
🔗 [GitHub Profile](https://github.com/shannu1438)

---

## Improvements

This project is built for learning and demonstration purposes.  can improve it by:
- Adding better preprocessing
- Deploying to cloud platforms (Render, Vercel, etc.)
