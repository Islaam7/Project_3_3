from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io, os

app = Flask(__name__)

# ====== إعدادات ======
MODEL_PATH = r"C:\\Users\\MS\\Desktop\\mohamed\\417_Ai\\project_1_3\\saved_model\\best_model.h5"
IMAGE_SIZE = (128, 128)   # لازم نفس حجم التدريب
TOP_K = 39                # نعرض أعلى 5 احتمالات

# ==== تحميل الموديل مرة واحدة عند بدء السيرفر ====
print("Loading model...")
model = load_model(MODEL_PATH)
model.make_predict_function()  # للحفاظ على الأداء في بعض البيئات
print("Model loaded.")

# ==== ضع أسماء الفئات هنا (عدلها حسب class_indices عندك) ====
# لو عندك ملف JSON للفئات يمكن قراءته بدل اللستة الثابتة.
class_names = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Background_without_leaves': 4, 'Blueberry___healthy': 5, 'Cherry___Powdery_mildew': 6, 'Cherry___healthy': 7, 'Corn___Cercospora_leaf_spot Gray_leaf_spot': 8, 'Corn___Common_rust': 9, 'Corn___Northern_Leaf_Blight': 10, 'Corn___healthy': 11, 'Grape___Black_rot': 12, 'Grape___Esca_(Black_Measles)': 13, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14, 'Grape___healthy': 15, 'Orange___Haunglongbing_(Citrus_greening)': 16, 'Peach___Bacterial_spot': 17, 'Peach___healthy': 18, 'Pepper,_bell___Bacterial_spot': 19, 'Pepper,_bell___healthy': 20, 'Potato___Early_blight': 21, 'Potato___Late_blight': 22, 'Potato___healthy': 23, 'Raspberry___healthy': 24, 'Soybean___healthy': 25, 'Squash___Powdery_mildew': 26, 'Strawberry___Leaf_scorch': 27, 'Strawberry___healthy': 28, 'Tomato___Bacterial_spot': 29, 'Tomato___Early_blight': 30, 'Tomato___Late_blight': 31, 'Tomato___Leaf_Mold': 32, 'Tomato___Septoria_leaf_spot': 33, 'Tomato___Spider_mites Two-spotted_spider_mite': 34, 'Tomato___Target_Spot': 35, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 36, 'Tomato___Tomato_mosaic_virus': 37, 'Tomato___healthy': 38}

# ===== صفحة الويب الرئيسية =====
@app.route("/")
def index():
    return render_template("index.html")

# ===== API للتنبؤ: تستقبل ملف صورة وترد JSON =====
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400

    # اقرأ الصورة
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "cannot read image", "details": str(e)}), 400

    # تحضير للصيغة اللي اتدرب عليها الموديل
    img = img.resize(IMAGE_SIZE)
    x = img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1, H, W, 3)

    # تنبؤ
    preds = model.predict(x)[0]   # احتمالات لكل كلاس
    # نظم أعلى نتائج
    top_indices = preds.argsort()[-TOP_K:][::-1]
    results = []
    for i in top_indices:
        results.append({
            "class_index": int(i),
            "class_name": list(class_names.keys())[list(class_names.values()).index(int(i))] if int(i) in class_names.values() else f"Class_{i}",
            "probability": float(preds[i])
        })

    # الأفضل إرسال top1 كحقل منفصل
    top1 = results[0] if results else None
    return jsonify({"predictions": results, "predicted_class": top1})

# ===== تشغيل محلي =====
if __name__ == "__main__":
    # لتشغيل على الشبكة المحلية استخدم host="0.0.0.0" وport حسب رغبتك
    app.run(host="0.0.0.0", port=5000, debug=True)
