import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
from io import BytesIO
import base64

class ImageSearch:
    # Load the pre-trained ResNet50 model
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
    # Define the image preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    IMAGE_DIR = "c:/temp/images/"
    FILE_LIST = "c:/temp/filelist"
    FEATURE_LIST = "c:/temp/features.pt"
    
    features_list = torch.load(FEATURE_LIST )
    with open(FILE_LIST, 'rb') as f:
        filelist = pickle.load(f)


    def __init__(self):
        print("ImageSearch init xxxxxxxxxxxxx")

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def getFeature(self,imgFile):
        image = Image.open(imgFile)
        image = ImageSearch.preprocess(image)
        image = image.unsqueeze(0).to(ImageSearch.device)
        with torch.no_grad():
            features = ImageSearch.model(image)

        return features

    def getFeatureFromBase64(self,base64String):
        f = BytesIO()
        f.write(base64.b64decode(base64String))
        f.seek(0)
        image = Image.open(f)
        image = ImageSearch.preprocess(image)
        image = image.unsqueeze(0).to(ImageSearch.device)
        with torch.no_grad():
            features = ImageSearch.model(image)

        return features

    def searchImagesFromFile(self, queryImage , topK=3):
        queryFeature = ImageSearch.getFeature(self,queryImage)
        result_files, score_list = ImageSearch.searchImages(self,queryFeature)
                
        return result_files, score_list

    def searchImagesFromBase64(self, queryImage , topK=5):
        queryFeature = ImageSearch.getFeatureFromBase64(self,queryImage)
        result_files, score_list = ImageSearch.searchImages(self,queryFeature)
        
        return result_files, score_list
    
    def searchImages(self, queryFeature , topK=3):
        score_list = []
        result_files = []
        for i, feature in enumerate(ImageSearch.features_list):
            score_t = ImageSearch.cos_sim(queryFeature, feature)
            score = score_t.item()
            if len(score_list) < topK :
                score_list.append(score)
                result_files.append(ImageSearch.filelist[i])
            else:
                if min(score_list) < score:                    
                    min_idx = score_list.index(min(score_list))
                    score_list[min_idx] = score
                    result_files[min_idx] = ImageSearch.filelist[i]
                
        # Return the top-k images and their scores
        return result_files, score_list