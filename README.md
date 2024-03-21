# CS766_Project

We provided a requirements.txt for recreating the virtual environment.
```
python3 -m venv cv
source ./cv/bin/activate
pip install -r requirements.txt
```

The dataset is the images directory, where no_mask is the origional images and with_mask is the origional images with masks. train_test_split was performed and train/test folders were generated. In these two folder the images from no_mask and with_mask are stitched together.

To preprocess the dataset, run
```
python preprocess.py
```

To train, run
```
python train.py
```

To do inference, run
```
python inference.py
```

Trained model can be found in this [link](https://drive.google.com/drive/folders/1OA2VwcP72DmbmBsy8bW5UjgKZYQ4ublk?usp=drive_link). To use this model, download the 'runs' directory along with the models in it and place this directory into the root of this git repo.

The output folder is the inference result based on images/test folder. We use the images without the mask to perform the inference and the images with mask are for reference.
