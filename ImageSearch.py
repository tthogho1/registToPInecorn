import torch
import clip
from PIL import Image

image_path = 'c:/temp/images/1365510055.jpg'
TEXT_LIST = ["town near a square","town near a sea", "town in europe", "town in asia"] 
# デバイスの指定
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデルの読み込み
model, preprocess = clip.load("ViT-L/14", device=device)

print("printing image path: " + image_path)

text = clip.tokenize(TEXT_LIST).to(device)

# 画像の読み込みと前処理
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

with torch.no_grad(): 
    # 画像、テキストのエンコード 
    image_features = model.encode_image(image)
    print (image_features.shape)
    print (image_features)
    text_features = model.encode_text(text) 
    print (text_features.shape)
    print (text_features)
    # 推論 
    logits_per_image, logits_per_text = model(image, text) 
    probs = logits_per_image.softmax(dim=-1).cpu().numpy() 


# 類似率の出力（テキスト毎に類似率を表示） 
for i in range(len(TEXT_LIST)): 
    rate = probs[0][i] 
    print(TEXT_LIST[i] + "---" + str(rate))
