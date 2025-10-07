
import numpy as np
import faiss
import joblib
import pickle
from sklearn.linear_model import LogisticRegression

# Paths
base_path = "."

# 1. Dummy MLP model (1280-dim input = 768 (BERT) + 512 (CLIP))
print("✅ Saving dummy MLP classifier...")
mlp_model = LogisticRegression(max_iter=300).fit(
    np.random.rand(30, 1280), np.random.randint(0, 3, 30)
)
joblib.dump(mlp_model, f"{base_path}/mlp_model.pkl")

# 2. FAISS Indexes
print("✅ Creating FAISS indexes...")
text_index = faiss.IndexFlatIP(768)
image_index = faiss.IndexFlatIP(512)

text_vectors = np.random.rand(30, 768).astype("float32")
image_vectors = np.random.rand(30, 512).astype("float32")

text_index.add(text_vectors)
image_index.add(image_vectors)

faiss.write_index(text_index, f"{base_path}/text_faiss.index")
faiss.write_index(image_index, f"{base_path}/image_faiss.index")

# 3. Cache files
print("✅ Creating text/image sentiment caches...")
text_cache = {i: int(np.random.choice([-1, 0, 1])) for i in range(30)}
image_cache = {i: int(np.random.choice([-1, 0, 1])) for i in range(30)}

with open(f"{base_path}/text_cache.pkl", "wb") as f:
    pickle.dump(text_cache, f)

with open(f"{base_path}/image_cache.pkl", "wb") as f:
    pickle.dump(image_cache, f)

print("🎉 All model and cache files saved.")
