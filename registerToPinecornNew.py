import os
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, CLIPProcessor, CLIPModel, CLIPTokenizer

# from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone
from pymongo import MongoClient
import shutil
import json

import configparser

conf = configparser.ConfigParser()

conf.read("./setting.conf")

driver_URL = conf.get("driver", "mongo_url")
image_folder = conf.get("image", "image_folder")
pinecorn_Index = conf.get("pinecone", "pinecone_index")
pinecorn_api_key = conf.get("pinecone", "pinecone_api_key")
clip_model = conf.get("clip", "model")
LOGFILE = "c:/temp/log.txt"

pc = Pinecone(api_key=pinecorn_api_key)
index = pc.Index(pinecorn_Index)


def get_model_info(model_ID, device):
    model = CLIPModel.from_pretrained(model_ID).to(device)
    processor = AutoProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return model, processor, tokenizer


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"

model, processor, tokenizer = get_model_info(model_ID, device)


def get_single_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    # normalize input embeddings
    text_embeddings = model.get_text_features(**inputs)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    # convert the embeddings to numpy array
    return text_embeddings.cpu().detach().numpy()


def get_single_image_embedding(my_image):
    image = processor(images=my_image, return_tensors="pt")
    embedding = model.get_image_features(**image).float()
    # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()


def writeLog(text):
    with open(LOGFILE, "a") as f:
        f.write(text)
        f.write("\n")


def createMetaInfo(webCamInfo):
    return {
        "webcamid": str(webCamInfo["webcam"]["webcamid"]),
        "status": webCamInfo["webcam"]["status"],
        "title": webCamInfo["webcam"]["title"],
        "country": webCamInfo["webcam"]["location"]["country"],
        "latitude": webCamInfo["webcam"]["location"]["latitude"],
        "longitude": webCamInfo["webcam"]["location"]["longitude"],
        "day": webCamInfo["webcam"]["player"]["day"],
        "images": str(webCamInfo["webcam"]["webcamid"]),
    }


with MongoClient(driver_URL) as client:
    webcamDb = client.webcamNew
    webCamCol = webcamDb.webcam

    # search all information
    for webCamInfo in webCamCol.find():
        # get webcamid from webCamInfo
        try:

            webcamId = webCamInfo["webcam"]["webcamid"]
            imageSvr = "https://bucket4image.s3.ap-northeast-1.amazonaws.com/"
            imageUrl = imageSvr + str(webcamId) + ".jpg"
            print(imageUrl)

            # Create image from the URL
            response = requests.get(imageUrl)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                # image.save(filename)
            else:
                print(f"Failed to retrieve image from {imageUrl}")
                continue

            imageFeature_np = get_single_image_embedding(image)
            imageEmbedding = imageFeature_np[0].tolist()
            # print(imgEmbedding)

            metaInfo = createMetaInfo(webCamInfo)
            metaInfoString = json.dumps(metaInfo)
            print(metaInfoString)

            upsert_response = index.upsert(
                vectors=[
                    {
                        "id": str(webcamId),
                        "values": imageEmbedding,
                        "metadata": metaInfo,
                    }
                ],
                namespace="webcamInfo",
            )

        except Exception as e:
            print(e)
            continue
