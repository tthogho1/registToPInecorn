from flask import Blueprint, request,jsonify
import json
import torch
import clip
from PIL import Image

from pinecone import Pinecone

import configparser
conf = configparser.ConfigParser()
conf.read('./setting.conf')

search_module = Blueprint('search_module', __name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

pinecone_Index = conf.get('pinecone', 'pinecone_index')
pinecone_api_key = conf.get('pinecone', 'pinecone_api_key')

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_Index)



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
    
    text = clip.tokenize(prompt).to(device)
    textFeature = model.encode_text(text) 
    embedding_np = textFeature.cpu().detach().numpy()
    embedding_list = embedding_np.tolist()
    print(embedding_np)
    
    resIndex = index.query(
        namespace="webcamInfo",
        vector=embedding_list,
        top_k=3,
        include_metadata=True
    )

    return json.dumps(resIndex.to_dict())


