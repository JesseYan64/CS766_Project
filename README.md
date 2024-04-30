# CS766_Project

We provided a requirements.txt for recreating the virtual environment.
```
python3 -m venv cv
source ./cv/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
```

The dataset is the data directory, where no_cover is the origional images, with_mask is the origional images with masks, and r_cover is the origional images with random covering. train_test_split was performed and train/test folders were generated. In these two folder the images from no_cover and with_mask are stitched together. train_r/test_r are from no_cover and r_cover.

To preprocess the dataset, run
```
python preprocess.py
```

To train Pix2Pix, run
```
python train_pix2pix.py
```

To train Pix2Pix on random covering, run
```
python train_pix2pix.py --dataset-path=./data/train_r --output=pix2pix_r
```

To train CEGAN, run
```
python train_cegan.py
```

To train CEGAN on random covering, run
```
python train_cegan.py --dataset-path=./data/train_r --output=cegan_r
```

To do inference, run
```
python inference.py
```
Note that you need to change parameters, including 'model', 'cegan' and 'path_to_test'.

You can find the dataset via this [link](https://drive.google.com/drive/folders/1B1QefmIljQ6Kr1rkIY2jF5YtAwCG1XB7?usp=sharing). It has no_cover, with_mask, r_cover, and train/test for these images.

Trained model can be found in this [link](https://drive.google.com/drive/folders/1OA2VwcP72DmbmBsy8bW5UjgKZYQ4ublk?usp=drive_link). To use this model, download the 'runs' directory along with the models in it and place this directory into the root of this git repo.

The output folder is the inference result based on data/test folder. We use the images with the mask to perform the inference and the images without the mask are for reference. You can find the output via this [link](https://drive.google.com/drive/folders/1Z7vLgt9IQHN2ofBcEFTLW_DSRurLcmRk?usp=sharing).
