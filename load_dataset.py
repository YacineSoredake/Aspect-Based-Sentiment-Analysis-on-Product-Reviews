import kagglehub

path = kagglehub.dataset_download(
    "charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis"
)
print("Path to dataset files:", path)
