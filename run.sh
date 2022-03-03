log_file=log_test.txt
sh_mode=1 # 0 for train and 1 for test
if [ $1 != '' ]; then
    sh_mode=$1
fi
echo 'sh_mode:' ${sh_mode}
cuda=2
echo 'using cuda:' $cuda
if [ ${sh_mode} == 0 ]; then
echo 'train!'
CUDA_VISIBLE_DEVICES=$cuda python main.py --gpus=1 --model_choice=use_img_clip --max_image_id=307 \
    --fix_text=false --fix_img=true --train_data_path=./data/train_pair.json --train_batch_size=8 \
    --add_emotion_task=false --add_predict_context_task=false --add_predict_img_label_task=false --add_ocr_info=false \
    --img_lr=5e-7 --other_lr=9e-6 --sent_num=0 --gradient_accumulation_steps=1 2>&1 | tee $log_file
elif [ ${sh_mode} == 1 ]; then
echo 'test!' | tee ${log_file}
for with_cand in false true ;
do 
    for test_path in  test_easy test_hard ;
    do
    echo $with_cand $test_path 2>&1 | tee -a $log_file
    CUDA_VISIBLE_DEVICES=$cuda python main.py --gpus=1 --max_image_id=307 --add_ocr_info=true --test_with_cand=${with_cand} \
        --sent_num=0 --valtest_batch_size=1 --ckpt_epoch=2 --add_emotion_task=true --add_predict_context_task=true --add_predict_img_label_task=true \
        --mode=test --test_data_path=./data/${test_path}.json --ckpt_path=./logs/clip/lightning_logs/version_135/checkpoints --ocr_path=./data/ocr_max10.json 2>&1 | tee -a $log_file
    done
done 
elif [ ${sh_mode} == 2 ]; then
echo 'predict img label'
CUDA_VISIBLE_DEVICES=$cuda python main.py --gpus=1 --model_choice=use_img_clip --max_image_id=307 \
    --sent_num=1 --valtest_batch_size=1 --ckpt_epoch=8 --test_with_cand=false --add_predict_img_label_task=true \
    --mode=predict_img_label --test_data_path=./data/test_easy.json --ckpt_path=./logs/clip/lightning_logs/version_90/checkpoints
elif [ ${sh_mode} == 3 ]; then
echo 'test sentence gradient'
test_path=test_easy
log_file=log_gradient_test_easy.txt
CUDA_VISIBLE_DEVICES=$cuda python main.py --gpus=1 --max_image_id=307 --add_ocr_info=true --test_with_cand=false \
        --sent_num=0 --valtest_batch_size=1 --ckpt_epoch=2 --add_emotion_task=true --add_predict_context_task=true --add_predict_img_label_task=true \
        --mode=test_gradient --test_data_path=./data/${test_path}.json --ckpt_path=./logs/clip/lightning_logs/version_135/checkpoints --ocr_path=./data/ocr_max10.json 2>&1 | tee -a $log_file

else
echo 'predict img emotion'
test_path=validation_pair
CUDA_VISIBLE_DEVICES=$cuda python main.py --gpus=1 --max_image_id=307 --add_ocr_info=true --test_with_cand=false \
        --sent_num=0 --valtest_batch_size=1 --ckpt_epoch=2 --add_emotion_task=true --add_predict_context_task=true --add_predict_img_label_task=true \
        --mode=predict_emotion --test_data_path=./data/${test_path}.json --ckpt_path=./logs/clip/lightning_logs/version_135/checkpoints --ocr_path=./data/ocr_max10.json 2>&1 | tee -a $log_file

fi

