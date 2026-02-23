import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

MODEL_PATH     = os.path.join(OUTPUT_DIR, "model.pth")
DATA_PATH      = os.path.join(OUTPUT_DIR, "data.pkl")
VEC_TEXT_PATH  = os.path.join(OUTPUT_DIR, "vec_text.pkl")
VEC_TITLE_PATH = os.path.join(OUTPUT_DIR, "vec_title.pkl")