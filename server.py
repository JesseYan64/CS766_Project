from flask import Flask, send_file,request
import inference

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/inference')
def inference_image():
    output_folder = "./outputs/server_output"
    path_to_test = './data/test_small'
    model_path = f'./runs/cegan/generator_1000.pt'

    inference.test(output_folder, path_to_test=path_to_test, cegan=True, model_path=model_path)

    return send_file('outputs/server_output/0.png', mimetype='image/jpeg')

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
