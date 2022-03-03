# Sticker-Selection
This repository contains the code for paper "Selecting Stickers in Open-Domain Dialogue through Multitask Learning" accepted to the Findings of ACL 2022.

## dependency
python>=3.6

`pip install -r requirements.txt`
## process data
1.download dstc10 data at https://github.com/lizekang/DSTC10-MOD/tree/main/data

2.put the downloaded data in the directory `dstc_data` (including train.json, validation.json, c_test_easy_task2.json, c_test_hard_task2.json). Note that train.json is renamed from the downloaded c_train.json.

3.put the downloaded data in the directory `data`(including img2id.json, meme_set).

4.`python create_data.py`
## train
`bash run.sh 0`

## test
`bash run.sh 1`
