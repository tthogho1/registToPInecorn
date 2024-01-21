import torch
import clip
import os
import json
import shutil
from PIL import Image
from pymongo import MongoClient
from pinecone import Pinecone

import configparser
conf = configparser.ConfigParser()

conf.read('./setting.conf')

driver_URL = conf.get('driver', 'mongo_url')
image_folder = conf.get('image', 'image_folder')
image_bkfolder = conf.get('image', 'image_bk_folder')
pinecorn_Index = conf.get('pinecone', 'pinecone_index')
pinecorn_api_key = conf.get('pinecone', 'pinecone_api_key')
clip_model = conf.get('clip', 'model')
LOGFILE = 'c:/temp/log.txt'

pc = Pinecone(api_key=pinecorn_api_key)
index = pc.Index(pinecorn_Index)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device)

def getImageFeatures(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        return features

def writeLog(text):
    with open(LOGFILE, 'a') as f:
        f.write(text)
        f.write('\n')

def createMetaInfo(webCamInfo):
        webcamMetaInfo = {}
        webcamMetaInfo["webcamid"] = str(webCamInfo["webcamid"])
        webcamMetaInfo["status"] = webCamInfo["status"]
        webcamMetaInfo["title"] = webCamInfo["title"]
        webcamMetaInfo["country"] = webCamInfo["location"]["country"]
        webcamMetaInfo["latitude"] = webCamInfo["location"]["latitude"]
        webcamMetaInfo["longitude"] = webCamInfo["location"]["longitude"]
        webcamMetaInfo["day"] = webCamInfo["player"]["day"]
        webcamMetaInfo["images"] = str(webCamInfo["webcamid"])
        
        return webcamMetaInfo

with MongoClient(driver_URL) as client:
    webcamDb = client.webcam
    webCamCol = webcamDb.webcam

    # search all information
    for webCamInfo in webCamCol.find():
        # get webcamid from webCamInfo
        try:
            webcamId = str(webCamInfo["webcamid"])
            filename = image_folder + webcamId + '.jpg'
            bkfilename = image_bkfolder + webcamId + '.jpg' 
            if not os.path.exists(filename):
                continue
            
            print(filename)
            #writeLog(filename)
            imageFeature = getImageFeatures(filename)
            print(imageFeature.shape)
            # convert tensor to vector
            imageFeature = imageFeature.reshape(768)
            
            metaInfo = createMetaInfo(webCamInfo)
            metaInfoString = json.dumps(metaInfo)
            print(metaInfoString)
            
            upsert_response = index.upsert(
                vectors=[
                    (webcamId, imageFeature, metaInfo),
                ],
                namespace="webcamInfo"
            )
            
            shutil.move(filename, bkfilename)
            
        except Exception as e:
            print(e)
            continue