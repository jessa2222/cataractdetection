from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(r'D:\catarac\catarac\catarac\static\uploads')

model = tf.keras.models.load_model(r'D:\catarac\catarac\catarac\katarak.h5')

def prepare_image(image_path, image_size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_filename = None
    accuracy = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_filename = filename
            file.save(img_path)
            image = prepare_image(img_path)
            prediction = model.predict(image)
            confidence = prediction[0][0]
            result = 'Cataract' if confidence > 0.5 else 'Normal'
            accuracy = confidence * 100 if result == 'Cataract' else (1 - confidence) * 100
    return render_template('index.html', result=result, accuracy=accuracy, img_filename=img_filename)

if __name__ == '__main__':
    app.run(debug=True)
