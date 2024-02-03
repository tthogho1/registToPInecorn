import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor,CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
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

# create image  list
#imageIds = ['1112092027','1169988637','1170267697','1203350464','1568408898']
imageIds =[]
# get imageid from image list
#  and create image path 
#  and upsert to pincone
for imageId in imageIds:
    image_path = image_folder + imageId + ".jpg"
    embedding_as_np = get_single_image_embedding(Image.open(image_path))

    upsert_response = index.upsert(
        vectors=[
            {
                'id' : imageId,
                'values': embedding_as_np[0].tolist(),
                'metadata': {},
            }
        ],
        namespace="webcamInfo"
    )

texts = ["town near a sea","town near a square","road", "mountain","building" ,"town in asia"] 
# 
# convert list to tensor and query
#
for text in texts:
    text_embedding_as_np = get_single_text_embedding(text)
    # Query from Pinecone
    resIndex = index.query(
        namespace="webcamInfo",
        vector=text_embedding_as_np.tolist(),
        top_k=1,
        include_metadata=True
    )
    print(f"{text} : {resIndex['matches'][0]['id']}  {resIndex['matches'][0]['score'].__str__()}")

# get imageid from image list
#  and create image path 
#  and convert to tensor and query from Pinecone
for imageId in imageIds:
    image_path = image_folder + imageId + ".jpg"
    embedding_as_np = get_single_image_embedding(Image.open(image_path))

    # Query from Pinecone
    resIndex = index.query(
        namespace="webcamInfo",
        vector=embedding_as_np[0].tolist(),
        top_k=2,
        include_metadata=True
    )
    print(f"{imageId} : {resIndex['matches'][1]['id']}  {resIndex['matches'][1]['score'].__str__()}")
