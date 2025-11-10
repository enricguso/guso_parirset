echo "B1"
CUDA_VISIBLE_DEVICES=1 python scnet/train.py --config_path conf/parirset_conf.yaml --save_path ../b1 --rir_mode b1
echo "D1"
CUDA_VISIBLE_DEVICES=1 python scnet/train.py --config_path conf/parirset_conf.yaml --save_path ../d1 --rir_mode d1
echo "D2"
CUDA_VISIBLE_DEVICES=1 python scnet/train.py --config_path conf/parirset_conf.yaml --save_path ../d2 --rir_mode d2
echo "D3"   
CUDA_VISIBLE_DEVICES=1 python scnet/train.py --config_path conf/parirset_conf.yaml --save_path ../d3 --rir_mode d3
echo "D4"
CUDA_VISIBLE_DEVICES=1 python scnet/train.py --config_path conf/parirset_conf.yaml --save_path ../d4 --rir_mode d4