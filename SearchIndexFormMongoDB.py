import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor,CLIPProcessor, CLIPModel, CLIPTokenizer
from pymongo import MongoClient
import shutil
import json

import configparser
conf = configparser.ConfigParser()

conf.read('./setting.conf')

driver_URL = conf.get('driver', 'mongo_url')
image_folder = conf.get('image', 'image_folder')
image_bkfolder = conf.get('image', 'image_bk_folder')
LOGFILE = 'c:/temp/log.txt'

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
    inputs = tokenizer(text, return_tensors = "pt")
    # normalize input embeddings
    text_embeddings = model.get_text_features(**inputs)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)     
    # convert the embeddings to numpy array
    return text_embeddings.cpu().detach().numpy()


def get_single_image_embedding(my_image):
    image = processor(images=my_image , return_tensors="pt")
    embedding = model.get_image_features(**image).float()
    # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()

def writeLog(text):
    with open(LOGFILE, 'a') as f:
        f.write(text)
        f.write('\n')

vector_database_field_name = 'embed' # define your embedding field name.
with MongoClient(driver_URL) as client:
    webcamDb = client.webcam
    webCamCol = webcamDb.webcam
    webcamId = '1010434248'
    filename = image_folder + webcamId + '.jpg'
    
    #imageFeature_np = get_single_image_embedding(Image.open(filename))
    text = "river"
    imageFeature_np = get_single_text_embedding(text)
    imageEmbedding = imageFeature_np[0].tolist()
    results = webCamCol.aggregate([
        {
            "$vectorSearch": {
                "queryVector": imageEmbedding,
                "path": vector_database_field_name,
                "numCandidates": 100, # this should be 10-20x the limit
                "limit": 10, # the number of documents to return in the results
                "index": 'imageindex', # the index name you used in the earlier step
            }
        },
        {
            "$match":{
                "location.country": 'United States'
            }
        }
    ])
    
    for doc in results:
        print(doc["webcamid"])
        print(doc["location"])
    