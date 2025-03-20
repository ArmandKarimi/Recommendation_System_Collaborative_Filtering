import kaggle

#authentification 
kaggle.api.authenticate()
print("Kaggle API authentication successful!")

#download data
kaggle.api.dataset_download_files('grouplens/movielens-20m-dataset', path='../data', unzip=True)

