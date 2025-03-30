import kagglehub

# Download latest version
path = kagglehub.dataset_download("eliasdabbas/web-server-access-logs")

print("Path to dataset files:", path)