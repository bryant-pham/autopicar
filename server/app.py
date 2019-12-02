from flask import Flask, request, Response
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from flask_cors import CORS
from time import gmtime, strftime

app = Flask(__name__)
CORS(app)
# model = torch.load('../models/model_blue_20191104-12-02.pt') # old model
model = torch.load('../models/model_carimages_NOVEERLEFT_20191106-20-46.pt') # no veer left model
# model = torch.load('../models/model_carimages_20191106-22-00.pt') # new model
model.eval()

STATES = {
    0: 'straight',
    1: 'right', 
    3: 'veer_right',
    2: 'veer_left'
}

@app.route('/')
def hello_world():
    print(model)
    return 'Hello, World!'

@app.route('/api/classify', methods=['POST'])
def classify():
    with torch.no_grad():
        model.eval()
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)

        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = transform(pil_img).float()
        img = img.unsqueeze(0)
        img = Variable(img)

        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)

        prediction_string = STATES[predicted.item()]
        dt = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        pil_img.save('captures/%s-%s-capture.png' % (dt, prediction_string))
        print('PREDICTION: ' + prediction_string)

        return str(predicted.item())


if __name__ == '__main__':
    app.run(host= '0.0.0.0')
