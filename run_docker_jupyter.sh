

data_dir=/home/mary/work/data_dir
cache_dir=/home/mary/work/data_dir/cache
code_dir=/home/mary/work/repos/drone_3d_dataset


sudo docker run --gpus all --runtime=nvidia -v $data_dir:/workspace/ -v $cache_dir:/home/user/.cache/ \
-v $code_dir:/code/ -p 7009:7007 -p 8888:8888 --rm -it drone_ds:1.0 bash -c "
pip install notebook \
&& pip install pycolmap==3.13 \
&& pip install opencv-python \
&& pip install scipy \
&& sudo apt update \
&& sudo apt install -y libimage-exiftool-perl \
&& jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
"