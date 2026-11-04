from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("CSTR-Edinburgh/vctk")
print("licence tags:", [t for t in info.tags if 'licen' in t])
for rev in (None, "refs/convert/parquet"):
    try:
        fs = api.list_repo_files("CSTR-Edinburgh/vctk", repo_type="dataset", revision=rev)
        print("REV", rev, len(fs)); print("\n".join(fs[:25]))
    except Exception as e: print("REV", rev, "error", e)
