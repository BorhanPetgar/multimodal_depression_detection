In `image_modality` you can infer on images. <br>
In `text_modality` you can infer texts which are on csv files. <br>
#### `image_modality`:
- `inference.py`: infer on images (input: image path)
- `inference_csv.py`: gets a csv file with columns of image, text and label and save the probs of the image inference in a csv file.
- `inference_test.py`: it's completely the same as the `inference_csv.py` whereas the input csv file is a csv file for test phase :-)
