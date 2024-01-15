from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import torch
import io
from PIL import Image
import httpx
import os
import numpy as np
import cv2
import matplotlib
import operator as op
app = FastAPI(debug=True)

model = torch.hub.load('/Users/bilge/PycharmProjects/YOLOProjects/yolov5', 'custom',
                       path='/Users/bilge/PycharmProjects/YOLOProjects/yolov5/runs/train/exp/weights/best.pt', source='local')
IMAGE_DIR = "/Users/bilge/Desktop"
os.makedirs(IMAGE_DIR, exist_ok=True)
@app.post("/detect/")
async def detect(request: Request):
    # Read the image file
    image_data = await request.body()
    image_filename = "received_image.jpg"  # You might want to generate a unique filename
    image_path = os.path.join(IMAGE_DIR, image_filename)

    # Save the image
    with open(image_path, "wb") as image_file:
        image_file.write(image_data)

    # Open the saved image for processing
    with open(image_path, "rb") as image_file:
        image = Image.open(io.BytesIO(image_file.read()))
    # Perform detection
    results = model(image)
    print(results)
    if len(results.xyxy[0]) == 1:
        prob = float(results.xyxy[0][0][4])  # Probability of the most confident detection
    else:
        prob = 0.0
    # Check if the detection is a cat with high confidence
    is_cat = prob > 0.5 and results.names[int(results.xyxy[0][0][5])] == 'cat'
    print("oki")
    print(prob)

    if is_cat:
        files = {'file': (image_filename, open(image_path, 'rb'), 'image/jpeg')}
        async with httpx.AsyncClient() as client:
            response = await client.post("http://192.168.122.76:8080/api/files/uploadFile", files=files)
            response.raise_for_status()
            return response.text


    water_image = cv2.imread(image_path)
    mask = np.zeros(water_image.shape[:2], np.uint8)

    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    rectangle = (200, 0, 110, 240)

    cv2.grabCut(water_image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    nparr = water_image * mask2[:, :, np.newaxis]
    edges = cv2.Canny(nparr, 100, 200)

    max_w = 0
    index = 0
    for i in range(10, len(edges) - 10):

        if op.countOf(edges[i], 255) > max_w:
            max_w = op.countOf(edges[i], 255)
            print(max_w)
            index = i

    perc = int((len(edges) - index) / (len(edges)))
    print(perc)
    cv2.imwrite('houghlines.jpg', edges)

    async with httpx.AsyncClient() as client:
        # Construct the URL with the query parameter
        url = f"http://192.168.122.76:8080/api/log/updateWater?number={perc}"

        # Send the GET request with the query parameter
        response = await client.post(url)

        # Check if the request was successful
        response.raise_for_status()

        # Return the response text
        return response.text




# Run the server using: uvicorn filename:app --reload


