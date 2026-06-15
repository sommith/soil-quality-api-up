from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. ສ້າງຕົວປ່ຽນ FastAPI
app = FastAPI(title="Soil Quality IoT API")

# 2. ໂຫຼດໂມເດວ Random Forest 
model = joblib.load('rf_model.pkl')

# 3. ສ້າງຕາຕະລາງແປງຄ່າຕົວເລກ (0, 1, 2, 3) ໃຫ້ເປັນຂໍ້ຄວາມ
class_mapping = {0: "Moderate", 1: "Excellent", 2: "Good", 3: "Poor""}

# 4. ກຳນົດຮູບແບບຂໍ້ມູນທີ່ IoT ຈະສົ່ງມາ (7 ຄ່າຫຼັກ)
class SoilInput(BaseModel):
    temperature: float
    humidity: float
    EC: float
    pH: float
    nitrogen: float
    phosphorus: float
    potassium: float

# 5. ສ້າງບ່ອນຮັບຂໍ້ມູນ (Endpoint) ສົ່ງຄ່າແບບ POST
@app.post("/predict")
def predict_soil(data: SoilInput):
    input_features = np.array([[ 
        data.temperature, data.humidity, data.EC, data.pH, 
        data.nitrogen, data.phosphorus, data.potassium 
    ]])
    prediction = model.predict(input_features)[0]
    prediction_label = class_mapping.get(int(prediction), "Unknown")
    return {"status": "success", "soil_quality": prediction_label}

# ຫນ້າທຳອິດສຳລັບ Test Server
@app.get("/")
def home():
    return {"message": "API is running successfully!"}
