# All dataset training Neuron Segmentation
nohup python tools/train.py configs/swin-sim-seg/1_mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sim_seg.py --gpu-ids 0 --work-dir results_all_data/09-19-2022/ > results_all_data/09-19-2022/output_all_data_seg.log &

