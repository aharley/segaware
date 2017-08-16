#!/bin/bash

NET_ID=seg
DATA_ROOT=~/datasets/VOC2012
#TRAIN_SET=voc_train_aug
TRAIN_SET=voc_trainval_aug
#TEST_SET=voc_val
TEST_SET=voc_test
DATASET=voc12
NUM_LABELS=21
#ITERS=100
#ITERS=1449
#ITERS=10582
ITERS=1456
#ITERS=7

ITER=5000

EXP=segaware
CAFFE_DIR=~/segaware/caffe
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe.bin

EMBS_LR1=1
EMBS_LR2=2
EMBS_LR3=2

MAIN_LR1=1
MAIN_LR2=2

FINAL_LR1=1
FINAL_LR2=2

CRF_LR1=0
CRF_LR2=0

SOLVER_MODE=GPU
GPU_ID="0"
CROP_SIZE=513
HALF_SIZE=257
BATCH_SIZE=1
LEARNING_RATE=0.001

# these are chosen empirically (but pad depends on the other two)
MASK_SIZE=3
MASK_DILATION=1
MASK_COVERAGE=`echo $((MASK_SIZE + (MASK_SIZE-1)*(MASK_DILATION-1)))`
MASK_PAD=`echo $((MASK_COVERAGE/2))`

MASK2_SIZE=3
MASK2_DILATION=2
MASK2_COVERAGE=`echo $((MASK_SIZE + (MASK_SIZE-1)*(MASK_DILATION-1)))`
MASK2_PAD=`echo $((MASK_COVERAGE/2))`

MASK3_SIZE=3
MASK3_DILATION=5
MASK3_COVERAGE=`echo $((MASK_SIZE + (MASK_SIZE-1)*(MASK_DILATION-1)))`
MASK3_PAD=`echo $((MASK_COVERAGE/2))`

# Create dirs
CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
echo $MODEL_DIR
FEATURE_DIR=${EXP}/features/${NET_ID}
mkdir -p ${FEATURE_DIR}/${TEST_SET}/mycrf
mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
#mkdir -p ${FEATURE_DIR}/${TEST_SET}/two
#mkdir -p ${FEATURE_DIR}/${TEST_SET}/four
#mkdir -p ${FEATURE_DIR}/${TEST_SET}/six
#mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc_fusion
echo $FEATURE_DIR
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

MODEL=segaware/model/${NET_ID}/train_voc_trainval_aug_iter_${ITER}.caffemodel

for pname in test; do
    sed "$(eval echo $(cat sub.sed))" \
        ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TEST_SET}.prototxt
done

CMD="${CAFFE_BIN} test \
         --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${GPU_ID} \
         --iterations=${ITERS}"

echo Running ${CMD} && ${CMD}
