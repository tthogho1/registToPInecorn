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
model_ID = "openai/clip-vit-large-patch32"
#model_ID = "ViT-L/14"

model, processor, tokenizer = get_model_info(model_ID, device)

def get_single_text_embedding(text): 
    inputs = tokenizer(text, return_tensors = "pt")
    # normalize input embeddings
    text_embeddings = model.get_text_features(**inputs)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True) 
    
    # convert the embeddings to numpy array
    return text_embeddings.cpu().detach().numpy()


def get_single_image_embedding(my_image):
    # image = processor(
	#	text = None,
	#	images = my_image,
	#	return_tensors="pt"
#		)["pixel_values"].to(device)
    image = processor(images=my_image , return_tensors="pt")
    embedding = model.get_image_features(**image).float()
    # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()

image_path = 'c:/temp/images/000000039769.jpg'
embedding_as_np = get_single_image_embedding(Image.open(image_path))
print(embedding_as_np[0].tolist())

upsert_response = index.upsert(
    vectors=[
        {
            'id' : "000000039769",
            'values': embedding_as_np[0].tolist(),
            'metadata': {},
        }
    ],
    namespace="webcamInfo"
)






text_embedding_as_np1 = get_single_text_embedding("town near a square")
text_embedding_as_np2 = get_single_text_embedding("river")
text_embedding_as_np3 = get_single_text_embedding("mountain")
text_embedding_as_np4 = get_single_text_embedding("town in europe")
text_embedding_as_np5 = get_single_text_embedding("cat")


# calculate cosine similarity with embedding_as_np and text_embedding_as_npi
cosine_similarity_score1 = cosine_similarity(embedding_as_np, text_embedding_as_np1)
print(f"town near a square : {cosine_similarity_score1}")

cosine_similarity_score2 = cosine_similarity(embedding_as_np, text_embedding_as_np2)
print(f"river : {cosine_similarity_score2}")

cosine_similarity_score3 = cosine_similarity(embedding_as_np, text_embedding_as_np3)
print(f"mountain : {cosine_similarity_score3}")

cosine_similarity_score4 = cosine_similarity(embedding_as_np, text_embedding_as_np4)
print(f"town in europe : {cosine_similarity_score4}")

cosine_similarity_score5 = cosine_similarity(embedding_as_np, text_embedding_as_np5)
print(f"cat : {cosine_similarity_score5}")



