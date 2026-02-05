# Thermal Inspection – Layer 0 (SIYI ZT30)

## Purpose

Layer 0 handles camera connection and thermal physics correction
(emissivity, distance, ambient temperature) using SIYI SDK.

## Setup (Laptop)

````bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Setup Jason
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Run
python main_pipeline.py



That’s enough.

---

## 4️⃣ `.gitignore` (VERY important)

Add this immediately:

```gitignore
venv/
__pycache__/
*.pyc
*.npy
*.log
data/
````
