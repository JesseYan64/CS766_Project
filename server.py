from flask import Flask, send_file,request
import inference_web
from datetime import datetime
import os
import base64
from PIL import Image
from io import BytesIO
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/inference' , methods=['POST'])
def inference_image():
    output_folder = "./outputs/server_output"
    path_to_test = './data/test_small'
    model_path = f'./runs/cegan/generator_1000.pt'

    img_temp_folder = './tmp/'
    current_time = datetime.now().time().strftime('%H_%M_%S')
    image_path = img_temp_folder + current_time 

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image_name = os.path.join(image_path, current_time + '.jpg')
    output_image_name = os.path.join(output_folder, current_time + '.jpg')

    # Assuming the base64 string is sent via form-data with the key 'photo'
    print('start paring data')
    data = request.form['image']
    # The data usually starts with a header like "data:image/png;base64,"
    # We need to split the string on ',' to remove this header
    header, encoded = data.split(',', 1)
    image_data = base64.b64decode(encoded)

    # Now we create an Image object and save it
    image = Image.open(BytesIO(image_data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(image_name)
    print("Photo saved locally")

    # Assuming inference_web.test is a function you've defined elsewhere
    inference_web.test(output_image_name, image_name, cegan=True, model_path=model_path)

    # Send the processed image back to the client
    return send_file(output_image_name, mimetype='image/jpeg')


@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        # Save the photo or perform further processing
        photo.save('uploaded_photo.jpg')
        return 'Photo uploaded successfully'
    return 'No photo uploaded'




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
