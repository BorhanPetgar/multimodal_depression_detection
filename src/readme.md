In `image_modality` you can infer on images. <br>
In `text_modality` you can infer texts which are in csv files. <br>
***
#### `multimodal`:
- `hyper.py`: This code is used for hyperparameter tuning to find the best weights for image and text modalities, after hyperparameter tuning, you can input the text modality weight in the last line of this script and evaluate your test csv file.
#### `image_modality`:
- `inference.py`: infer on images (input: image path)
- `inference_csv.py`: gets a csv file with columns of image, text and label and save the probs of the image inference in a csv file.
- `inference_test.py`: it's completely the same as the `inference_csv.py` whereas the input csv file is a csv file for test phase :-)
#### `text_modality`:
- `inference.py`: infer on csv file for testing
- `text_model.py`: it contains training and inference method

***
### ToDo
- collect data and put it here: `/home/borhan/Desktop/multimodal_depression_detection/data/test/text/text.csv`
- inference image modality with `src/image_modality/inference_test.py`
- inference text modality with `src/text_modality/inference.py`
- merge the 2 csv files
- run `hyper.py`
- 
