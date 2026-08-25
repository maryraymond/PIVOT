# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

data_dir=<data_directory_path>
cache_dir=<cache_directory_path>
code_dir=<PIVOT_repo_local_directory_path>

docker run --gpus all --runtime=nvidia -v $data_dir:/data/ -v $code_dir:/workspace/ -v $code_dir:/code/ -v $cache_dir:/home/user/.cache/ \
-p 8080:8080 --rm -it pivot:1.0.0 
