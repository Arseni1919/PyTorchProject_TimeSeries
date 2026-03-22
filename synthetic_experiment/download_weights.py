from huggingface_hub import snapshot_download

snapshot_download(repo_id="amazon/chronos-2", local_dir="./chronos2_weights")
print("Done.")