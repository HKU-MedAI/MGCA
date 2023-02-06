CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset object_cxr --data_pct 0.01 --learning_rate 2.5e-4 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset object_cxr --data_pct 0.1 --learning_rate 5e-4
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset object_cxr --data_pct 1 --learning_rate 5e-4
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 0.01 --learning_rate 2e-4 --batch_size 8
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 0.1 --learning_rate 5e-4
CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --dataset rsna --data_pct 1 --learning_rate 5e-4