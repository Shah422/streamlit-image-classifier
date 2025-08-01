
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, class_names):
    inputs = preprocess_image(image)
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    top3_probs, top3_idxs = torch.topk(probs, k=3)
    return top3_probs[0].tolist(), [class_names[i] for i in top3_idxs[0].tolist()]

# Streamlit UI
st.title("🧠 Image Classifier")

uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    class_names = load_class_names()
    probs, labels = predict(image, model, class_names)

    st.markdown(f"### 🔍 Top Prediction: `{labels[0]}` ({probs[0]*100:.2f}%)")
    st.subheader("📊 Top 3 Predictions:")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob*100:.2f}%")
