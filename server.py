from flask import Flask
import inference

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/inference')
def inference_image():
    output_folder = "./output"
    path_test = './test'
    model_path = f'../runs/cegan/generator_1000.pt'

    return "generated"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
