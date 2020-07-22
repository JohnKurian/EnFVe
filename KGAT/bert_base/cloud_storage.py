import requests

url = 'https://storage.googleapis.com/kgat_saved_model/vocab.txt'
r = requests.get(url, allow_redirects=True)
open('EnFVe/KGAT/bert_base/vocab.txt', 'wb').write(r.content)

url = 'https://storage.googleapis.com/kgat_saved_model/bert_config.json'
r = requests.get(url, allow_redirects=True)
open('EnFVe/KGAT/bert_base/bert_config.json', 'wb').write(r.content)

url = 'https://storage.googleapis.com/kgat_saved_model/pytorch_model.bin'
r = requests.get(url, allow_redirects=True)
open('EnFVe/KGAT/bert_base/pytorch_model.bin', 'wb').write(r.content)

url = 'https://storage.googleapis.com/kgat_saved_model/model.best.pt'
r = requests.get(url, allow_redirects=True)
open('EnFVe/KGAT/model.best.pt', 'wb').write(r.content)

'''
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/content/KernelGAT/kgat/JADGS-687c611bc710.json"

from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.get_bucket("kgat_saved_model")

from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

download_blob("kgat_saved_model", "vocab.txt", "EnFVe/KGAT/bert_base/vocab.txt")
download_blob("kgat_saved_model", "model.best.pt", "EnFVe/KGAT/model.best.pt")
download_blob("kgat_saved_model", "pytorch_model.bin", "EnFVe/KGAT/bert_base/pytorch_model.bin")
download_blob("kgat_saved_model", "vocab.txt", "EnFVe/KGAT/bert_base/vocab.txt")
'''
