source activate im2txt
bazel build -c opt //im2txt:get_feats
bazel-bin/im2txt/get_feats \
    --checkpoint_path=/data/home/v-lixxue/github/models/research/im2txt/im2txt_5M/model.ckpt-5000000 \
    --input_files=/mnt/coco-tf/raw-data/val2014/* \
    --feats_file=cap_ft_val_feats
bazel-bin/im2txt/get_feats \
    --checkpoint_path=/data/home/v-lixxue/github/models/research/im2txt/im2txt_5M/model.ckpt-5000000 \
    --input_files=/mnt/coco-tf/raw-data/train2014/*  \
    --feats_file=cap_ft_train_feats
