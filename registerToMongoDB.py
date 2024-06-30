import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor,CLIPProcessor, CLIPModel, CLIPTokenizer
from pymongo import MongoClient, ASCENDING
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



def process_batch(webCamInfoList : list):
    for webCamInfo in webCamInfoList:
        try:
            webcamId = str(webCamInfo["webcamid"])
            filename = image_folder + webcamId + '.jpg'
            bkfilename = image_bkfolder + webcamId + '.jpg' 
            if not os.path.exists(filename):
                continue
            
            print(filename)
            imageFeature_np = get_single_image_embedding(Image.open(filename))
            imageEmbedding = imageFeature_np[0].tolist()

            if vector_database_field_name not in webCamInfo:
                webCamInfo[vector_database_field_name] = imageEmbedding
            
            webCamCol.replace_one({'_id': webCamInfo['_id']}, webCamInfo)                        
            shutil.move(filename, bkfilename)
            
        except Exception as e:
            print(e)

vector_database_field_name = 'embed' # define your embedding field name.
with MongoClient(driver_URL) as client:
    webcamDb = client.webcam
    webCamCol = webcamDb.webcam
    
    last_id = None
    query = {}
    batch_size = 1000
    while True:
        # get webCamInfoList batch size = 1000 from webCamCol, the last '_id'
        if last_id:
            query['_id'] = {'$gt': last_id}        
        
        webCamInfoList = list(webCamCol.find(query).sort('_id', ASCENDING).limit(batch_size))
        if not webCamInfoList:
            break
        
        process_batch(webCamInfoList)
        last_id = webCamInfoList[-1]['_id']