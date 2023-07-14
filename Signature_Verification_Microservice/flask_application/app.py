from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
MODEL_PATH = './model/signature-model-008.h5'
model = tf.keras.models.load_model(MODEL_PATH,custom_objects={'F':tf.keras.models.Model})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        select = request.form.get('cust_id')
        
        f = request.files['upload_file']
        img_path = ("./static/uploads/" + f.filename)
        f.save(img_path)
        upsig=(f.filename)

        image1=("./static/reference_signatures/%s.png"%select)
        image2=("./static/uploaded_signatures/%s"%upsig)
        print(image1)
        print(image2)
        img_h, img_w = 155, 220

    # Preprocessing the image
        img1 = cv2.imread(image1, 0)
          
        img2 = cv2.imread(image2, 0)
        img1 = cv2.resize(img1, (img_w, img_h))
        
        img2 = cv2.resize(img2, (img_w, img_h))
        img1 = np.array(img1, dtype=np.float64)
       
        img2 = np.array(img2, dtype=np.float64)

        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                pixel = img1.item(i, j)
                if pixel > 200:
                    img1[i][j] = 255
                else:
                    img1[i][j] = 0
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                pixel = img2.item(i, j)
                if pixel > 200:
                    img2[i][j] = 255
                else:
                    img2[i][j] = 0
        
        img1 /= 255
        img2 /= 255
        img1 = img1[..., np.newaxis]
        
        # print(np.isnan(img1).any())
        img2 = img2[..., np.newaxis]
        # print(np.isnan(img2).any())
        img1 = img1.reshape(1,155,220,1)
        img2 = img2.reshape(1, 155, 220, 1)
        result = model.predict([img1, img2])
        prob = result[0][0]
        # image1 = '.'+image1
        # image2 = '.'+image2
        optimal_threshold = 0.3270535
        if prob < optimal_threshold:
            str1 = "Its a Forged Signature"
        else:
            str1 = "Its a Genuine Signature"
    
        return render_template("index.html", probability=prob, prediction=str(str1), ref_sig=image1, upl_sig=image2)
               

if __name__ == '__main__':
    app.run(debug = True)