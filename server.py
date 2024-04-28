from flask import Flask, send_file,request
import inference_web
from datetime import datetime
import os



app = Flask(__name__)

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

    os.mkdir(image_path)


    image_name = image_path + '/' +  current_time + '.jpg'
    output_image_name = output_folder +  '/' +  current_time + '.jpg'



    print("start downloading phot to locally")
    photo = request.files['photo']
    # Save the photo or perform further processing
    photo.save(image_name)
    print("photo saved")

    inference_web.test(output_image_name, image_name, cegan=True, model_path=model_path)

    return send_file('', mimetype='image/jpeg')

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
