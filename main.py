import io
import json
import os
from flask import Flask, render_template, request, jsonify

import cv2
from datetime import datetime
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
model.eval()    

img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

def transform_image(infile):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg

def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)

    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]

    return prediction_idx, class_name


@app.route('/',methods=["GET","POST"])
def predict():
    class_name = 'Unknown'
    class_id = 'unknown'
    img_dir = "static/imgs/"
    if request.method == 'GET':img_path=None
    elif request.method=='POST':
        #### POSTにより受け取った画像を読む
        file = request.files['img']
        #### POSTにより受け取った画像を「imgs/」に保存する
        stream = request.files['img']
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        #### 現在時刻を名前として「imgs/」に保存する
        dt_now =datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = img_dir + dt_now + ".jpg"
        cv2.imwrite(img_path, img)

        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
                    
    #### 推論結果をHTMLに渡す
    return render_template('index.html', class_name=class_name, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=80)