

data_dir=/home/mary/work/data_dir
cache_dir=/home/mary/work/data_dir/cache
code_dir=/home/mary/work/repos/drone_3d_dataset

sudo docker run --gpus all --runtime=nvidia -v $data_dir:/workspace/ -v $cache_dir:/home/user/.cache/ \
-v $code_dir:/code/ -p 7008:7007 --rm -it drone_ds:1.0                      