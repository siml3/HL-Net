gpu_id=0,1
port=10086
gpu_num=2
output_dir=""

PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --master_port ${port} --nproc_per_node=${gpu_num} tools/relation_test_net.py --config-file "${output_dir}/config.yml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True OUTPUT_DIR "${output_dir}" TEST.IMS_PER_BATCH ${gpu_num} DTYPE "float16" GLOVE_DIR glove