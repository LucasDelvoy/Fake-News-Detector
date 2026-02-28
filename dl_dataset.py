import kagglehub

# Download latest version
path = kagglehub.dataset_download("amar5693/fake-and-real-news-dataset-4k")

print("Path to dataset files:", path)