# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.


data_dir=<data_directory_path>
cache_dir=<cache_directory_path>
code_dir=<PIVOT_repo_local_directory_path>


sudo docker run --gpus all --runtime=nvidia -v $code_dir:/workspace/ -v $code_dir:/code/ -v $data_dir:/data/ -v $cache_dir:/home/user/.cache/ \
 -p 7009:7007 -p 8889:8888 -p 8080:8080 --rm -it pivot:0.1   bash -c " \
pip install matplotlib && \
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root "

