import io
import json
import torch
from transformers import AutoProcessor,CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from pinecone import Pinecone
from flask import Blueprint, request,jsonify

import configparser
conf = configparser.ConfigParser()
conf.read('./setting.conf')

search_module = Blueprint('search_module', __name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"

def get_model_info(model_ID, device):
	model = CLIPModel.from_pretrained(model_ID).to(device)
	processor = AutoProcessor.from_pretrained(model_ID)
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
	return model, processor, tokenizer

model, processor, tokenizer = get_model_info(model_ID, device)

pinecone_Index = conf.get('pinecone', 'pinecone_index')
pinecone_api_key = conf.get('pinecone', 'pinecone_api_key')

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_Index)


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


@search_module.route('/test', methods=['GET'])
def test_metal():
    return 'Metal'

@search_module.route('/testPost', methods=['POST'])
def test_post():
    prompt = request.form.get('name')
    print(f"prompt : {prompt}")
    return 'MetalPost'


@search_module.route('/searchByText', methods=['POST'])
def searchByText():
    prompt = request.form.get('prompt')
    print(f"prompt : {prompt}")
    
    text_embedding_as_np = get_single_text_embedding(prompt)
    # Query from Pinecone
    resIndex = index.query(
        namespace="webcamInfo",
        vector=text_embedding_as_np.tolist(),
        top_k=3,
        include_metadata=True
    )

    print(resIndex.to_dict())
    return json.dumps(resIndex.to_dict())

@search_module.route('/searchByImage', methods=['POST'])
def searchByImage():
    #print(request)[]
    # get uploaded file
    file = request.files.get('file')
    # set to Image format
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))
    image_embedding_as_np = get_single_image_embedding(image)
    # Query from Pinecone
    resIndex = index.query(
        namespace="webcamInfo",
        vector=image_embedding_as_np[0].tolist(),
        top_k=3,
        include_metadata=True
    )
    print(resIndex.to_dict())
    return json.dumps(resIndex.to_dict())

