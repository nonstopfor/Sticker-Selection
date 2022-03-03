import json
import os
from utils import try_create_dir
from tqdm import tqdm
from copy import deepcopy
from random import seed, randint, choice


def get_id2name(path='./data/id2img.json'):
    id2name = {}
    with open(path, encoding='utf-8') as f:
        id2img = json.load(f)
        for id, img in id2img.items():
            id = int(id)
            img = img[:img.find('.')]
            if img[0].isdigit():
                name = img
            else:
                if img[-1].isdigit():
                    name = img[:-1]
                else:
                    name = img
            # print(name)
            id2name[id] = name
    return id2name


def get_train_ids(name='./dstc_data/train.json'):
    with open(name, encoding='utf-8') as f:
        a = json.load(f)

    train_ids = set()
    for k, v in a.items():
        for d in v:
            id = d.get('img_id', None)
            if id:
                id = int(id)
                train_ids.add(id)

    return list(train_ids)


def create_data(in_file_name, out_file_name, pair=False):
    seed(2021)
    res = []
    max_id = 0
    id2name = get_id2name()
    train_ids = get_train_ids()
    tot = 0
    with open(in_file_name, encoding='utf-8') as in_f:
        a = json.load(in_f)
        for idx, (k, v) in tqdm(enumerate(a.items())):
            dialog = []
            for i, r in enumerate(v):
                speaker = r['speaker_id']
                if 'txt' not in r:
                    print(r)
                text = r['txt']
                text = speaker + text
                img_id = r.get('img_id', None)
                # dialog.append(speaker + text)
                if img_id is not None:
                    img_id = int(img_id)
                emotion_id = r.get('emotion_id', None)
                # assert not (img_id is not None and emotion_id is None), v
                if img_id is not None and emotion_id is None:
                    emotion_id = -100
                if img_id:
                    d = {
                        'text': text,
                        'img_id': img_id,
                        'img_label': id2name.get(img_id, None),
                        'emotion_id': emotion_id
                    }
                    dialog.append(d)
                else:
                    d = {
                        'text': text,
                        'img_id': None,
                        'img_label': None,
                        'emotion_id': emotion_id
                    }
                    dialog.append(d)
                if i > 0 and img_id:
                    max_id = max(max_id, img_id)
                    outd = deepcopy(dialog)
                    if pair:
                        pos_id = outd[-1]['img_id']
                        neg_id = pos_id
                        while neg_id == pos_id:
                            # neg_id = randint(0, 300)
                            neg_id = choice(train_ids)
                        outd[-1]['neg_img_id'] = neg_id
                        outd[-1]['neg_img_label'] = id2name.get(neg_id)
                    tot += 1
                    if img_id is not None and emotion_id is None:
                        continue
                    res.append({'dialog': outd})
    print(f"{out_file_name}, len:{len(res)}, original total_len:{tot}, max img id:{max_id}")

    with open(out_file_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def create_test_data(in_file_name, out_file_name, candidate=False):
    seed(2021)
    res = []
    max_id = 0
    id2name = get_id2name()
    cnt = 0
    with open(in_file_name, encoding='utf-8') as in_f:
        a = json.load(in_f)
        for v in tqdm(a):
            dialog = []
            for i, r in enumerate(v['history']):
                speaker = r['speaker_id']
                if 'txt' not in r:
                    print(r)
                text = r['txt']
                text = speaker + text
                img_id = r.get('img_id', None)
                emotion_id = r.get('emotion_id', None)
                # dialog.append(speaker + text)
                if img_id:
                    img_id = int(img_id)
                if img_id:
                    d = {
                        'text': text,
                        'img_id': img_id,
                        'img_label': id2name.get(img_id, None),
                        'emotion_id': emotion_id
                    }
                    dialog.append(d)
                else:
                    d = {
                        'text': text,
                        'img_id': None,
                        'img_label': None
                    }
                    dialog.append(d)
            ans = v['answer']
            assert ans['speaker_id'] == v['history'][-1]['speaker_id']
            img_id = int(ans['img_id'])
            max_id = max(max_id, img_id)
            img_name = id2name.get(img_id, None)
            dialog[-1]['img_id'] = img_id
            dialog[-1]['img_label'] = img_name
            outd = deepcopy(dialog)
            if candidate:
                cand = v['candidate']['set']
                cand = [int(t) for t in cand]
                res.append({'dialog': outd, 'cand': cand, 'idx': cnt})
                cnt += 1
            else:
                res.append({'dialog': outd, 'idx': cnt})
                cnt += 1

    print(f"{out_file_name}, max img id:{max_id}")

    with open(out_file_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def create_img_data():
    # os.system("cp ../../data/MOD-Dataset/supplementary/img2id.json ./data")
    # os.system("cp -r ../../data/MOD-Dataset/supplementary/meme_set ./data")
    with open('./data/img2id.json', 'r', encoding='utf-8') as f:
        a = json.load(f)
    with open('./data/img2id.json', 'w', encoding='utf-8') as f:
        json.dump(a, f, indent=2, ensure_ascii=False)
    id2img = {}
    imgs = os.listdir('./data/meme_set')
    # print(imgs)
    print(len(imgs))  # one img not in dict, ignore that img
    imgs = set(imgs)

    with open('./data/img2id.json', encoding='utf-8') as f:
        img2id = json.load(f)
        for img, id in img2id.items():
            id2img[id] = img
            if img not in imgs:
                print(imgs)
                print(img)
            assert img in imgs

    with open('./data/id2img.json', 'w', encoding='utf-8') as f:
        json.dump(id2img, f, indent=2, ensure_ascii=False)

    with open('./data/id2name.json', 'w', encoding='utf-8') as f:
        id2name = get_id2name()
        print(id2name)
        json.dump(id2name, f, indent=2, ensure_ascii=False)


def split_test_into_seen_unseen(test_name, seen_name, unseen_name):
    train_ids = set(get_train_ids())

    with open(test_name, encoding='utf-8') as f:
        a = json.load(f)

    seen = []
    unseen = []
    for d in a:
        dialog = d['dialog']
        id = dialog[-1]['img_id']
        if id in train_ids:
            seen.append(d)
        else:
            unseen.append(d)

    print(
        f"total test len:{len(a)}, seen len:{len(seen)}, unseen len:{len(unseen)}")

    with open(seen_name, 'w', encoding='utf-8') as outf:
        json.dump(seen, outf, indent=1, ensure_ascii=False)

    with open(unseen_name, 'w', encoding='utf-8') as outf:
        json.dump(unseen, outf, indent=1, ensure_ascii=False)


def chunk_data(in_file_name, out_file_name, num=8):
    with open(in_file_name, encoding='utf-8') as f:
        a = json.load(f)

    res = a[:num]
    repeat_num = 1000
    import itertools
    x = list(itertools.repeat(res, repeat_num))
    res = list(itertools.chain(*x))
    print(res[0])
    print(len(res))
    with open(out_file_name, 'w', encoding='utf-8') as out_f:
        json.dump(res, out_f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    
    in_data_dir = './dstc_data'
    out_dir = './data'
    try_create_dir(out_dir)
    create_img_data()
    for split in ['train', 'validation']:
        create_data(f"{os.path.join(in_data_dir, split)}.json",
                    f"{os.path.join(out_dir, split)}_pair.json", pair=True)

    create_test_data(f"{os.path.join(in_data_dir, 'c_test_easy_task2')}.json",
                     f"{os.path.join(out_dir, 'test_easy')}.json", candidate=True)
    create_test_data(f"{os.path.join(in_data_dir, 'c_test_hard_task2')}.json",
                     f"{os.path.join(out_dir, 'test_hard')}.json", candidate=True)
