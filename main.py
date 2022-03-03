from logging import log
from posixpath import join
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, HfArgumentParser, BertTokenizer
import transformers
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers.utils.dummy_pt_objects import LogitsProcessor
from utils import get_logger, top_filtering, try_create_dir, Timer
import os
import json
from copy import deepcopy
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim

from metrics import MyAccuracy
import clip

logger = get_logger(__name__, 'logger_unseen.txt')
# logger = get_logger(__name__)


class BertModel(BertForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)


class Model(torch.nn.Module):
    def __init__(self, args):
        # config.num_labels = 307
        # super().__init__(config)
        super().__init__()
        # self.segment_embedding = torch.nn.Embedding(2, config.hidden_size) # 0: speaker1 1: speaker2
        self.args = args
        logger.info(self.args.model_choice)
        self.temperature = torch.nn.Parameter(torch.tensor(args.init_temp))
        self.bert = BertModel.from_pretrained(args.bert_pretrain_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_pretrain_path)
        special_tokens_dict = {
            'additional_special_tokens': ['[speaker1]', '[speaker2]']
        }
        self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.bert_tokenizer))
        if self.args.model_choice == 'use_clip_repo':
            logger.info('use clip repo')
            self.clip_model, self.clip_process = clip.load("ViT-B/32")
        elif self.args.model_choice == 'use_img_id':
            logger.info('use img id')
            self.text_clip = SentenceTransformer(
                'clip-ViT-B-32-multilingual-v1')
            self.img_embedding_layer = torch.nn.Embedding(
                args.max_image_id, 512)
        else:
            logger.info('use image clip')
            self.img_clip = SentenceTransformer('clip-ViT-B-32')
            # self.text_clip = SentenceTransformer(
            #     'clip-ViT-B-32-multilingual-v1')
            if args.fix_text:
                self.bert.eval()
                for p in self.bert.parameters():
                    p.requires_grad = False
            if args.fix_img:
                self.img_clip.eval()
                for p in self.img_clip.parameters():
                    p.requires_grad = False

        # self.text_clip = SentenceTransformer('clip-ViT-B-32')
        self.text_ff = torch.nn.Linear(768, 768)
        self.img_ff = torch.nn.Linear(512, 768)
        if args.add_emotion_task:
            self.emotion_head = torch.nn.Linear(768, args.max_emotion_id)
        if args.add_predict_context_task or args.add_predict_img_label_task:
            self.mlm_head = torch.nn.Linear(768, len(self.bert_tokenizer))
        # self.text_clip_ff = torch.nn.Linear(512, 512)
        with open(self.args.ocr_path, encoding='utf-8') as f:
            a = json.load(f)

        self.id2ocr = {}
        for k, v in a.items():
            self.id2ocr[int(k)] = v['ocr']

        self.id2name = {}
        with open('./data/id2name.json', encoding='utf-8') as f:
            a = json.load(f)
            for k, v in a.items():
                self.id2name[int(k)] = v

        self.predict_id2ocr = {}

    def get_image_obj(self, id):
        if id in self.id2img:
            return self.id2img[id]

        img_file_name = self.id2imgpath[id]
        img_dir = self.args.img_dir
        path = os.path.join(img_dir, img_file_name)
        from img_utils import pick_single_frame
        with open(path, 'rb') as fin:
            data = fin.read()
        img_obj = self.img_process(pick_single_frame(data))
        if self.args.mode != 'pretrain':
            self.id2img[id] = img_obj
        return img_obj

    def prepare_for_test(self):
        logger.info("prepare_for_test!")
        with torch.no_grad():
            if self.args.model_choice == 'use_clip_repo':
                img_objs = []
                for id in range(self.args.max_image_id):
                    img_obj = self.id2img[id]
                    img_objs.append(img_obj)
                device = next(self.clip_model.parameters()).device
                img = torch.stack(img_objs, dim=0).to(device)
                self.clip_model.eval()
                self.all_img_embs = self.clip_model.encode_image(img)

            elif self.args.model_choice == 'use_img_id':
                img_ids = list(range(self.args.max_image_id))
                self.all_img_embs = self.img_embedding_layer(torch.tensor(
                    img_ids, dtype=torch.long, device=self.text_clip.device))

            else:
                img_objs = []
                for id in range(self.args.max_image_id):
                    img_obj = self.get_image_obj(id)
                    img_objs.append(img_obj)
                img_tokens = self.img_clip.tokenize(img_objs)

                img_tokens = batch_to_device(
                    img_tokens, self.img_clip._target_device)
                self.img_clip.eval()
                self.all_img_embs = self.img_clip.forward(
                    img_tokens)['sentence_embedding']

    def get_emb_by_imgids(self, img_ids):
        img_objs = []
        for i, id in enumerate(img_ids):
            img_obj = self.get_image_obj(id)
            img_objs.append(img_obj)
        img_tokens = self.img_clip.tokenize(img_objs)

        img_tokens = batch_to_device(
            img_tokens, self.img_clip._target_device)

        img_emb = self.img_clip.forward(
            img_tokens)['sentence_embedding']
        return img_emb

    def get_input_output_imglabel_by_imgid(self, img_id):
        max_len = self.args.max_img_label_mask_num
        self.tokenizer = self.bert_tokenizer
        mask_id = self.tokenizer.mask_token_id
        cls_id = self.tokenizer.cls_token_id
        mask_predict_inputs = [mask_id] * max_len
        ocr = self.id2ocr[img_id]
        if ocr != '':
            ids = self.tokenizer.encode(ocr, add_special_tokens=False)
            # mask_predict_outputs = [mask_id] * \
            #     (max_len - len(ids) - 1) + [cls_id] + ids
            mask_predict_outputs = [-100] * \
                (max_len - len(ids) - 1) + [cls_id] + ids
            cls_inputs = [mask_id] * (max_len - len(ids)) + ids
        else:
            mask_predict_outputs = [-100] * max_len
            cls_inputs = [mask_id] * max_len
        return mask_predict_inputs, mask_predict_outputs, cls_inputs

    def check_has_ocr(self, img_id):
        ocr = self.id2ocr[img_id]
        if ocr == '':
            return False
        return True

    def update_predict_ocr(self, img_id, img_label, logits=None):
        if img_id not in self.predict_id2ocr:
            self.predict_id2ocr[img_id] = Counter()

        self.predict_id2ocr[img_id][img_label] += 1
        logger.debug(
            f"img_id: {img_id}, true img label: {self.id2name[img_id]}, pred img_label: {img_label}")
        if logits is not None:
            logits = logits.squeeze(0)
            _, indices = torch.topk(logits, k=5, dim=-1)
            for idx in indices:
                logger.debug(
                    f"tokens:{self.tokenizer.convert_ids_to_tokens(idx)}")

    def predict_img_label(self, input_ids, attention_mask, batch_idx=1, img_ids=None):
        device = input_ids.device
        img_emb = self.get_emb_by_imgids(img_ids)
        mask_predict_inputs_ts = []
        assert len(img_ids) == 1
        if self.check_has_ocr(img_ids[0]):
            return
        for img_id in img_ids:
            mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                img_id)
            mask_predict_inputs_ts.append(mask_predict_inputs)
        mask_predict_inputs_ts = torch.tensor(
            mask_predict_inputs_ts, device=device, dtype=torch.long)
        mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            mask_predict_inputs_ts)
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        batch_size = input_ids.size(0)

        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        # logger.debug(f"{text_emb.size()}, {mask_predict_inputs_emb.size()}, {img_emb.size()}, {sep_emb.size()}")
        mask_input_emb = torch.cat(
            [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

        extra_len = mask_input_emb.size(1) - text_emb.size(1)
        ones_mask2 = torch.ones(batch_size, extra_len, device=device)
        attention_mask = torch.cat(
            [attention_mask, ones_mask2], dim=1)

        token_type_ids = torch.zeros_like(
            attention_mask, dtype=torch.long)
        token_type_ids[:, -extra_len:] = 1

        mask_res = self.bert(inputs_embeds=mask_input_emb,
                             attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
        mask_hidden_states = mask_res.hidden_states
        start = text_emb.size(1)
        end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

        mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
        logits = self.mlm_head(mask_hidden_states)
        # logger.debug(logits.size())
        tokenizer = self.bert_tokenizer
        ids = logits.argmax(-1)[0]
        # res = tokenizer(ids[0])
        # logger.debug(res)
        generated_ids = None
        cls_id = tokenizer.cls_token_id
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == cls_id:
                if i == len(ids) - 1:
                    break
                generated_ids = ids[i+1:]
                break

        if generated_ids is not None:
            img_label = ''.join(tokenizer.convert_ids_to_tokens(generated_ids))
            self.update_predict_ocr(img_ids[0], img_label, logits)

    def test_img_emotion(self, input_ids, attention_mask, img_ids):
        img_emb = self.get_emb_by_imgids(img_ids)
        device = attention_mask.device
        batch_size = img_emb.size(0)
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # logger.debug(text_emb.size())
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        assert self.args.add_ocr_info is True
        if self.args.add_ocr_info:
            # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
            # mask_predict_inputs_ts = []
            # mask_predict_outputs_ts = []
            cls_inputs_ts = []

            for img_id in img_ids:
                mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                    img_id)
                # mask_predict_inputs_ts.append(mask_predict_inputs)
                # mask_predict_outputs_ts.append(mask_predict_outputs)
                cls_inputs_ts.append(cls_inputs)

            # mask_predict_inputs_ts = torch.tensor(
            #     mask_predict_inputs_ts, device=device, dtype=torch.long)
            # mask_predict_outputs_ts = torch.tensor(
            #     mask_predict_outputs_ts, device=device, dtype=torch.long)
            cls_inputs_ts = torch.tensor(
                cls_inputs_ts, device=device, dtype=torch.long)

            # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            #     mask_predict_inputs_ts)
            cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                cls_inputs_ts)

            input_emb = torch.cat(
                [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format

            extra_len = input_emb.size(1) - text_emb.size(1)
            ones_mask2 = torch.ones(batch_size, extra_len, device=device)
            addocr_attention_mask = torch.cat(
                [attention_mask, ones_mask2], dim=1)
            labels = torch.ones(
                batch_size, dtype=torch.long, device=device)

            token_type_ids = torch.zeros_like(
                addocr_attention_mask, dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
            hidden_states = res.hidden_states  # len=13, embedding + 12 layer
            hidden_states = hidden_states[-1][:, -2, :]
            logits = self.emotion_head(hidden_states)
            return logits.argmax(-1)

    def test_gradient(self, input_ids, attention_mask, img_ids):
        img_emb = self.get_emb_by_imgids(img_ids)
        device = attention_mask.device
        batch_size = img_emb.size(0)
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # logger.debug(text_emb.size())
        text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        # print(type(self.bert))
        assert self.args.add_ocr_info is True
        if self.args.add_ocr_info:
            # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
            # mask_predict_inputs_ts = []
            # mask_predict_outputs_ts = []
            cls_inputs_ts = []

            for img_id in img_ids:
                mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                    img_id)
                # mask_predict_inputs_ts.append(mask_predict_inputs)
                # mask_predict_outputs_ts.append(mask_predict_outputs)
                cls_inputs_ts.append(cls_inputs)

            # mask_predict_inputs_ts = torch.tensor(
            #     mask_predict_inputs_ts, device=device, dtype=torch.long)
            # mask_predict_outputs_ts = torch.tensor(
            #     mask_predict_outputs_ts, device=device, dtype=torch.long)
            cls_inputs_ts = torch.tensor(
                cls_inputs_ts, device=device, dtype=torch.long)

            # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
            #     mask_predict_inputs_ts)
            cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                cls_inputs_ts)

            input_emb = torch.cat(
                [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format

            extra_len = input_emb.size(1) - text_emb.size(1)
            ones_mask2 = torch.ones(batch_size, extra_len, device=device)
            addocr_attention_mask = torch.cat(
                [attention_mask, ones_mask2], dim=1)
            labels = torch.ones(
                batch_size, dtype=torch.long, device=device)

            token_type_ids = torch.zeros_like(
                addocr_attention_mask, dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
            logits = res.logits
            success = (logits.argmax(-1) == labels).item()
            hidden_states = res.hidden_states  # len=13, embedding + 12 layer
            # hidden_states = hidden_states[-1][:, -2, :]
            embeddings = hidden_states[0]
            embeddings.retain_grad()
            loss = res.loss
            loss.backward()
            # print(embeddings.size())
            # print(embeddings.grad)
            # print(embeddings.grad.size())
            # ret_grad = embeddings.grad.detach().clone()
            # print(torch.autograd.grad(loss, embeddings, retain_graph=True))
            # logits = self.emotion_head(hidden_states)
            # return logits.argmax(-1)
            self.zero_grad() # 不会清空中间变量的grad
            # for name, v in self.named_parameters():
            #     print(name)
            # self.bert.zero_grad()
            # print(embeddings.grad)
            # print(embeddings.grad.size())
            return embeddings.grad, success
            

    def forward(self, input_ids, attention_mask, batch_idx=1, neg_img_ids=None, img_ids=None, emotion_ids=None, img_tokens=None, neg_img_tokens=None,
                test=False, cands=None, mask_context_input_ids=None, mask_context_output_ids=None, mask_context_attention_mask=None):

        if self.args.mode == 'predict_img_label':
            return self.predict_img_label(input_ids=input_ids, attention_mask=attention_mask, batch_idx=batch_idx, img_ids=img_ids)

        if self.args.mode == 'predict_emotion':
            return self.test_img_emotion(input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids)

        if self.args.mode == 'test_gradient':
            return self.test_gradient(input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids)

        if not test:
            img_emb = self.get_emb_by_imgids(img_ids)
            neg_img_emb = self.get_emb_by_imgids(neg_img_ids)
        else:
            if cands:
                assert len(cands) == 1
                cands = cands[0]
                img_emb = self.all_img_embs[cands]
                if self.args.add_ocr_info:
                    cls_inputs_ts = []

                    for img_id in cands:
                        mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                            img_id)
                        cls_inputs_ts.append(cls_inputs)
                    cls_inputs_ts = torch.tensor(
                        cls_inputs_ts, device=self.all_img_embs.device, dtype=torch.long)
                    cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                        cls_inputs_ts)
            else:
                img_emb = self.all_img_embs
                if self.args.add_ocr_info:
                    cls_inputs_ts = []

                    for img_id in range(self.args.max_image_id):
                        mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                            img_id)
                        cls_inputs_ts.append(cls_inputs)
                    cls_inputs_ts = torch.tensor(
                        cls_inputs_ts, device=self.all_img_embs.device, dtype=torch.long)
                    cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                        cls_inputs_ts)
        # logger.info(f'unique img ids num:{len(list(set(img_ids)))}')

        if not test:
            device = attention_mask.device
            batch_size = img_emb.size(0)
            sep_emb = self.bert.bert.embeddings.word_embeddings(
                torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            # logger.debug(text_emb.size())
            text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
            img_emb = self.img_ff(img_emb).unsqueeze(1)
            neg_img_emb = self.img_ff(neg_img_emb).unsqueeze(1)
            if self.args.add_ocr_info:
                # neg_mask_predict_inputs, neg_mask_predict_outputs, neg_cls_inputs = self.get_input_output_imglabel_by_imgid(neg_img_id)
                # mask_predict_inputs_ts = []
                # mask_predict_outputs_ts = []
                cls_inputs_ts = []
                neg_cls_inputs_ts = []

                for img_id in img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    # mask_predict_inputs_ts.append(mask_predict_inputs)
                    # mask_predict_outputs_ts.append(mask_predict_outputs)
                    cls_inputs_ts.append(cls_inputs)

                for img_id in neg_img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    neg_cls_inputs_ts.append(cls_inputs)

                # mask_predict_inputs_ts = torch.tensor(
                #     mask_predict_inputs_ts, device=device, dtype=torch.long)
                # mask_predict_outputs_ts = torch.tensor(
                #     mask_predict_outputs_ts, device=device, dtype=torch.long)
                cls_inputs_ts = torch.tensor(
                    cls_inputs_ts, device=device, dtype=torch.long)
                neg_cls_inputs_ts = torch.tensor(
                    neg_cls_inputs_ts, device=device, dtype=torch.long)

                # mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                #     mask_predict_inputs_ts)
                cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    cls_inputs_ts)
                neg_cls_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    neg_cls_inputs_ts)

                input_emb = torch.cat(
                    [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)  # format

                extra_len = input_emb.size(1) - text_emb.size(1)
                ones_mask2 = torch.ones(batch_size, extra_len, device=device)
                addocr_attention_mask = torch.cat(
                    [attention_mask, ones_mask2], dim=1)
                labels = torch.ones(
                    batch_size, dtype=torch.long, device=device)
                neg_labels = torch.zeros(
                    batch_size, dtype=torch.long, device=device)
                token_type_ids = torch.zeros_like(
                    addocr_attention_mask, dtype=torch.long)
                token_type_ids[:, -extra_len:] = 1
                neg_input_emb = torch.cat(
                    [text_emb, neg_cls_inputs_emb, neg_img_emb, sep_emb], dim=1)
                res = self.bert(inputs_embeds=input_emb,
                                attention_mask=addocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                loss = res.loss
                neg_res = self.bert(inputs_embeds=neg_input_emb,
                                    attention_mask=addocr_attention_mask, labels=neg_labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                neg_loss = neg_res.loss
                cls_loss = (loss + neg_loss) / 2
                # if self.args.add_predict_img_label_task:
                #     mask_input_emb = torch.cat(
                #         [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

                #     mask_res = self.bert(inputs_embeds=mask_input_emb,
                #                         attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                #     mask_hidden_states = mask_res.hidden_states
                #     start = text_emb.size(1)
                #     end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

                #     mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
                #     logits = self.mlm_head(mask_hidden_states)
                #     # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                #     # labels = mask_predict_outputs_ts[:, 1:].contiguous()
                #     labels = mask_predict_outputs_ts
                #     loss_fct = torch.nn.CrossEntropyLoss()
                #     mask_loss = loss_fct(
                #         logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                #     final_loss = cls_loss + self.args.label_weight * mask_loss
                # else:
                final_loss = cls_loss

            else:
                input_emb = torch.cat(
                    [text_emb, img_emb, sep_emb], dim=1)
                ones_mask2 = torch.ones(batch_size, 2, device=device)
                noaddocr_attention_mask = torch.cat(
                    [attention_mask, ones_mask2], dim=1)
                labels = torch.ones(
                    batch_size, dtype=torch.long, device=device)
                neg_labels = torch.zeros(
                    batch_size, dtype=torch.long, device=device)
                token_type_ids = torch.zeros_like(
                    noaddocr_attention_mask, dtype=torch.long)
                token_type_ids[:, -2:] = 1
                # logger.debug(token_type_ids.dtype)
                res = self.bert(inputs_embeds=input_emb,
                                attention_mask=noaddocr_attention_mask, labels=labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                loss = res.loss
                neg_input_emb = torch.cat(
                    [text_emb, neg_img_emb, sep_emb], dim=1)
                neg_res = self.bert(inputs_embeds=neg_input_emb,
                                    attention_mask=noaddocr_attention_mask, labels=neg_labels, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                neg_loss = neg_res.loss
                cls_loss = (loss + neg_loss) / 2
                final_loss = cls_loss

            if self.args.add_predict_img_label_task:
                mask_predict_inputs_ts = []
                mask_predict_outputs_ts = []

                for img_id in img_ids:
                    mask_predict_inputs, mask_predict_outputs, cls_inputs = self.get_input_output_imglabel_by_imgid(
                        img_id)
                    mask_predict_inputs_ts.append(mask_predict_inputs)
                    mask_predict_outputs_ts.append(mask_predict_outputs)

                mask_predict_inputs_ts = torch.tensor(
                    mask_predict_inputs_ts, device=device, dtype=torch.long)
                mask_predict_outputs_ts = torch.tensor(
                    mask_predict_outputs_ts, device=device, dtype=torch.long)

                mask_predict_inputs_emb = self.bert.bert.embeddings.word_embeddings(
                    mask_predict_inputs_ts)
                mask_input_emb = torch.cat(
                    [text_emb, mask_predict_inputs_emb, img_emb, sep_emb], dim=1)

                predlabel_extra_len = mask_input_emb.size(1) - text_emb.size(1)
                predlabel_ones_mask2 = torch.ones(
                    batch_size, predlabel_extra_len, device=device)
                predlabel_attention_mask = torch.cat(
                    [attention_mask, predlabel_ones_mask2], dim=1)

                token_type_ids = torch.zeros_like(
                    predlabel_attention_mask, dtype=torch.long)
                token_type_ids[:, -predlabel_extra_len:] = 1

                mask_res = self.bert(inputs_embeds=mask_input_emb,
                                     attention_mask=predlabel_attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
                mask_hidden_states = mask_res.hidden_states
                start = text_emb.size(1)
                end = text_emb.size(1) + mask_predict_inputs_emb.size(1)

                mask_hidden_states = mask_hidden_states[-1][:, start:end, :]
                logits = self.mlm_head(mask_hidden_states)
                # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                # labels = mask_predict_outputs_ts[:, 1:].contiguous()
                labels = mask_predict_outputs_ts
                loss_fct = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fct(
                    logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                final_loss = final_loss + self.args.label_weight * mask_loss

            if self.args.add_emotion_task:
                hidden_states = res.hidden_states  # len=13, embedding + 12 layer
                hidden_states = hidden_states[-1][:, -2, :]
                logits = self.emotion_head(hidden_states)
                emotion_labels = torch.tensor(
                    emotion_ids, dtype=torch.long, device=device)
                loss_fct = torch.nn.CrossEntropyLoss()
                emotion_loss = loss_fct(logits, emotion_labels)
                if batch_idx % 1000 == 0:
                    logger.debug(f'{cls_loss}, {emotion_loss}')
                # return cls_loss + 0.1 * emotion_loss,
                final_loss = final_loss + self.args.emotion_weight * emotion_loss

            if self.args.add_predict_context_task:
                mask_context_text_emb = self.bert.bert.embeddings.word_embeddings(
                    mask_context_input_ids)
                if self.args.add_ocr_info:
                    mask_context_input_emb = torch.cat(
                        [mask_context_text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)
                    mask_context_attention_mask = torch.cat(
                        [mask_context_attention_mask, ones_mask2], dim=1)
                    mask_context_token_type_ids = torch.zeros_like(
                        mask_context_attention_mask, dtype=torch.long)
                    mask_context_token_type_ids[:, -extra_len:] = 1
                else:
                    mask_context_input_emb = torch.cat(
                        [mask_context_text_emb, img_emb, sep_emb], dim=1)
                    mask_context_attention_mask = torch.cat(
                        [mask_context_attention_mask, ones_mask2], dim=1)
                    mask_context_token_type_ids = torch.zeros_like(
                        mask_context_attention_mask, dtype=torch.long)
                    mask_context_token_type_ids[:, -2:] = 1

                mask_context_res = self.bert(inputs_embeds=mask_context_input_emb,
                                             attention_mask=mask_context_attention_mask, token_type_ids=mask_context_token_type_ids, return_dict=True, output_hidden_states=True)
                # len=13, embedding + 12 layer
                mask_context_hidden_states = mask_context_res.hidden_states
                mask_context_hidden_states = mask_context_hidden_states[-1][:, :mask_context_output_ids.size(
                    1), :]
                logits = self.mlm_head(mask_context_hidden_states)
                # shifted_prediction_scores = logits[:, :-1, :].contiguous()
                # labels = mask_context_output_ids[:, 1:].contiguous()
                labels = mask_context_output_ids
                loss_fct = torch.nn.CrossEntropyLoss()
                mask_context_loss = loss_fct(
                    logits.view(-1, len(self.bert_tokenizer)), labels.view(-1))
                final_loss = final_loss + self.args.ctx_weight * mask_context_loss
            if batch_idx % 1000 == 0:
                mask_loss = mask_loss if 'mask_loss' in dir() else None
                emotion_loss = emotion_loss if 'emotion_loss' in dir() else None
                mask_context_loss = mask_context_loss if 'mask_context_loss' in dir() else None
                logger.debug(
                    f"cls_loss:{cls_loss}, mask_loss:{mask_loss}, emotion_loss:{emotion_loss}, mask_context_loss:{mask_context_loss}")
            return final_loss,
        else:
            device = attention_mask.device
            batch_size = input_ids.size(0)
            assert batch_size == 1
            img_num = img_emb.size(0)
            sep_emb = self.bert.bert.embeddings.word_embeddings(
                torch.tensor(self.bert_tokenizer.sep_token_id, device=device, dtype=torch.long)).unsqueeze(0).unsqueeze(0).repeat(img_num, 1, 1)
            # logger.debug(text_emb.size())
            text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
            text_emb = text_emb.repeat(img_num, 1, 1)
            img_emb = self.img_ff(img_emb).unsqueeze(1)
            if self.args.add_ocr_info:
                input_emb = torch.cat(
                    [text_emb, cls_inputs_emb, img_emb, sep_emb], dim=1)
            else:
                input_emb = torch.cat(
                    [text_emb, img_emb, sep_emb], dim=1)
            extra_len = input_emb.size(1) - text_emb.size(1)

            token_type_ids = torch.zeros(size=(input_emb.size(
                0), input_emb.size(1)), device=device, dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            token_type_ids=token_type_ids, return_dict=True)
            # logger.debug(res.logits.size())
            logits = res.logits[:, 1].unsqueeze(0)
            labels = torch.tensor(img_ids, dtype=torch.long, device=device)
            return logits, labels, cands

    def prepare_imgs(self, args):
        self.args = args
        if self.args.model_choice == 'use_img_id':
            return
        img_dir = args.img_dir
        self.id2img = {}
        if self.args.model_choice == 'use_clip_repo':
            self.img_process = self.clip_process
        else:
            self.img_process = self.img_clip._first_module().preprocess
        logger.info("prepare img objs")
        self.id2imgpath = {}

        with open(args.id2img_path, encoding='utf-8') as f:
            id2img = json.load(f)
            for id, img in tqdm(id2img.items()):
                id = int(id)
                self.id2imgpath[id] = img

        # if args.fix_clip:
        #     logger.info('fix clip!')
        #     for p in self.img_clip.parameters():
        #         p.requires_grad = False

        # if self.args.mode != 'gen':
        #     assert self.text_clip.training == True
        #     assert self.img_clip.training == True

    def get_input_embeddings_with_segment(self, input_ids, segment_ids):
        # position_ids will be computed by the model itself even if the inputs_embeds is given
        # position_ids = torch.arange(input_ids.size(-1), device=input_ids.device).unsqueeze(0)
        # logger.debug(f"{self.transformer.wte(input_ids).size()}, {self.transformer.wpe(position_ids).size()}, {self.segment_embedding(segment_ids).size()}")
        img_mask = (segment_ids == 2)
        text_mask = (segment_ids != 2)
        text_embedding = self.bert.embeddings.word_embeddings(
            input_ids) * (text_mask.unsqueeze(-1))
        img_pos = img_mask.nonzero(as_tuple=True)
        img_ids = input_ids[img_pos]
        img_objs = []
        for i, id in enumerate(img_ids):
            img_obj = self.id2img[id.item()]
            img_objs.append(img_obj)

        if img_objs:

            img_tokens = self.img_clip.tokenize(img_objs)
            img_tokens = batch_to_device(
                img_tokens, self.img_clip._target_device)
            # img_emb = self.img_clip.encode(img_objs, convert_to_tensor=True)
            # img_emb.requires_grad = True
            # print(img_emb.size())
            # img_emb = img_emb.to(input_ids.device)
            img_emb = self.img_clip.forward(img_tokens)['sentence_embedding']
            if self.args.mode == 'gen':
                assert self.img_clip.training == False
            emb = self.img_ff(img_emb)
            # text_embedding[img_pos[0][i], img_pos[1][i]] = emb
            # logger.debug(f"emb: {emb.requires_grad}")
            zeros = torch.zeros_like(text_embedding, device=input_ids.device)
            zeros[img_pos] = emb
            # text_embedding[img_pos] = emb
            # logger.debug(f"zeros:{zeros.requires_grad}")
            text_embedding = text_embedding + zeros

        # img_emb = self.img_clip.encode(img_objs, convert_to_tensor=True)
        # # print(img_emb.size())
        # img_emb = img_emb.to(input_ids.device)
        # emb = self.img_ff(img_emb)
        # # text_embedding[img_pos[0][i], img_pos[1][i]] = emb
        # text_embedding[img_pos] = emb

        return text_embedding + self.segment_embedding(
            segment_ids)


class PLDataset(Dataset):
    def __init__(self, path, mode, args, tokenizer):
        with open(path, encoding='utf-8') as f:
            self.data = json.load(f)
        # self.tokenizer = tokenizer
        self.mode = mode
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def random_mask(self, text_ids, return_ts):
        # random mask for MLM
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        mask_id = self.tokenizer.mask_token_id
        l = len(self.tokenizer)
        for r, i in zip(rands, text_ids):
            # if i == helper.sep_id or i == helper.cls_id:
            #     input_ids.append(i)
            #     output_ids.append(-100)
            #     continue
            if r < 0.15 * 0.8:
                input_ids.append(mask_id)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(l))
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(-100)
        if return_ts:
            # logger.debug(f"input_ids: {input_ids}, output_ids: {output_ids}")
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
                output_ids, dtype=torch.long)
        else:
            return input_ids, output_ids

    def __getitem__(self, index):
        dialog = self.data[index]['dialog']
        sent_num = self.args.sent_num
        if sent_num == 0:
            selected_dialog = dialog
        else:
            selected_dialog = dialog[-sent_num:]
        sent = ''
        img_id = None

        def pad_none(x, l=5):
            return x + [None] * (l - len(x))
        for i, d in enumerate(selected_dialog):
            if i == len(selected_dialog) - 1:
                # s, img_id, img_word, neg_img_id, neg_img_word = pad_none(d, 5)
                s = d.get('text', None)
                img_id = d.get('img_id', None)
                img_word = d.get('img_label', None)
                neg_img_id = d.get('neg_img_id', None)
                neg_img_word = d.get('neg_img_label', None)
                emotion_id = d.get('emotion_id', None)
            else:
                # s, _, _ = d
                s = d['text']
            sent += s
        # sent, img_id, img_word, neg_img_id, neg_img_word = dialog[-1]
        # img_id = int(img_id)
        # sent = sent.replace('[speaker1]', '').replace('[speaker2]', '')

        if self.args.add_predict_context_task:
            sent_ids = self.tokenizer.encode(sent)
            # logger.debug(sent_ids)
            mask_context_input_ids, mask_context_output_ids = self.random_mask(
                sent_ids, return_ts=True)
            # logger.debug(mask_context_input_ids)
#
            mask_context_attention_mask = torch.ones(
                len(mask_context_input_ids), dtype=torch.float)
        else:
            mask_context_input_ids, mask_context_output_ids = None, None
            mask_context_attention_mask = None

        if self.args.test_with_cand:
            cand = [id for id in self.data[index]['cand']]

        else:
            cand = None

        res = {
            'sent': sent,
            'img_word': img_word,
            'img_id': img_id,
            'neg_img_word': neg_img_word,
            'neg_img_id': neg_img_id,
            'emotion_id': emotion_id,
            'cand': cand,
            'mask_context_output_ids': mask_context_output_ids,
            'mask_context_input_ids': mask_context_input_ids,
            'mask_context_attention_mask': mask_context_attention_mask
        }
        return res
        # return {k:v for k, v in res.items() if v is not None}

        if not self.args.test_with_cand:
            return {
                'sent': sent,
                'img_word': img_word,
                'img_id': img_id,
                'neg_img_word': neg_img_word,
                'neg_img_id': neg_img_id,
                'emotion_id': emotion_id,
            }
        else:
            return {
                'sent': sent,
                'img_word': img_word,
                'img_id': img_id,
                'neg_img_word': neg_img_word,
                'neg_img_id': neg_img_id,
                'cand': [id for id in self.data[index]['cand']]
            }


class PLDataLoader(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data_path = args.train_data_path
        self.val_data_path = args.val_data_path
        self.test_data_path = args.test_data_path
        self.train_batch_size = args.train_batch_size
        self.valtest_batch_size = args.valtest_batch_size
        self.gen_data_path = args.gen_data_path
        self.gen_batch_size = args.gen_batch_size
        self.args = args

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            logger.info('begin get data')

            self.train_dataset = PLDataset(self.train_data_path,
                                           self.args.mode, self.args, self.tokenizer)
            if self.args.val_data_path:
                self.val_dataset = PLDataset(
                    self.val_data_path, self.args.mode, self.args, self.tokenizer)
                logger.info('has val!')
            else:
                logger.info('no val!')
            logger.info('finish get data')

        if stage == 'test' or stage is None:
            logger.info('begin get data')
            if self.args.mode != 'gen':
                self.test_dataset = PLDataset(
                    self.test_data_path, self.args.mode, self.args, self.tokenizer)
            else:
                self.test_dataset = PLDataset(
                    self.gen_data_path, self.args.mode, self.args, self.tokenizer)
            logger.info('finish get data')

    def collate_fn(self, batch):
        # logger.debug(batch)
        sents = [d['sent'] for d in batch]
        # img_words = [d['img_word'] for d in batch]
        img_ids = [d['img_id'] for d in batch]
        # neg_img_words = [d['neg_img_word'] for d in batch]
        neg_img_ids = [d['neg_img_id'] for d in batch]
        chunk = 490
        res = self.tokenizer(sents, return_tensors='pt',
                             padding=True, truncation=True, max_length=chunk)
        input_ids = res['input_ids']
        attention_mask = res['attention_mask']
        if batch[0]['cand'] is not None:
            cands = [d['cand'] for d in batch]
        else:
            cands = None

        if batch[0]['emotion_id'] is not None:
            emotion_ids = [d['emotion_id'] for d in batch]
        else:
            emotion_ids = None

        if batch[0]['mask_context_input_ids'] is not None:
            # chunk = 500
            mask_context_input_ids = pad_sequence(
                [d['mask_context_input_ids'][-chunk:] for d in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            mask_context_output_ids = pad_sequence(
                [d['mask_context_output_ids'][-chunk:] for d in batch], batch_first=True, padding_value=-100)
            mask_context_attention_mask = pad_sequence(
                [d['mask_context_attention_mask'][-chunk:] for d in batch], batch_first=True, padding_value=0)
        else:
            mask_context_input_ids = None
            mask_context_output_ids = None
            mask_context_attention_mask = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'img_ids': img_ids,
            'neg_img_ids': neg_img_ids,
            'cands': cands,
            'emotion_ids': emotion_ids,
            'mask_context_input_ids': mask_context_input_ids,
            'mask_context_output_ids': mask_context_output_ids,
            'mask_context_attention_mask': mask_context_attention_mask,
        }

        if self.args.test_with_cand:
            cands = [d['cand'] for d in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'img_ids': img_ids,
                'neg_img_ids': neg_img_ids,
                'cands': cands
            }
        else:
            emotion_ids = [d['emotion_id'] for d in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'img_ids': img_ids,
                'neg_img_ids': neg_img_ids,
                'emotion_ids': emotion_ids
            }

    def train_dataloader(self, ):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        # return None
        if self.args.val_data_path:
            return DataLoader(self.val_dataset,
                              batch_size=self.valtest_batch_size,
                              num_workers=self.args.num_workers,
                              pin_memory=True,
                              shuffle=False,
                              collate_fn=self.collate_fn)
        else:
            return None

    def test_dataloader(self):
        batch_size = self.valtest_batch_size if self.args.mode != 'gen' else self.gen_batch_size
        return DataLoader(self.test_dataset,
                          batch_size=batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=self.collate_fn)


class PLModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
        # logger.info(f"tokenizer type:{type(self.tokenizer)}")(
        self.model = Model(args)
        # special_tokens_dict = {
        #     # 'pad_token': '[PAD]',
        #     'additional_special_tokens': ['[speaker1]', '[speaker2]', '[img]']
        # }
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # logger.info(
        #     f"tokenizer add special tokens dict: {special_tokens_dict}")
        # self.model.resize_token_embeddings(len(self.tokenizer))
        # self.lr = args.lr
        # self.acc = MyAccuracy(device=torch.device("cuda"))
        # self.train_acc5 = MyAccuracy()
        # self.train_acc30 = MyAccuracy()
        # self.train_acc90 = MyAccuracy()
        self.train_acc = MyAccuracy()
        self.valtest_acc5 = MyAccuracy()
        self.valtest_acc30 = MyAccuracy()
        self.valtest_acc90 = MyAccuracy()
        self.valtest_map = MyAccuracy()
        self.args = args
        self.model.prepare_imgs(args)
        self.id2name = {}
        with open('./data/id2name.json', encoding='utf-8') as f:
            a = json.load(f)
            for k, v in a.items():
                self.id2name[int(k)] = v

    def run_model_from_batch(self, batch, batch_idx, test=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # img_tokens = batch['img_tokens']
        img_ids = batch['img_ids']
        # neg_img_tokens = batch['neg_img_tokens']
        neg_img_ids = batch['neg_img_ids']
        cands = batch.get('cands', None)
        emotion_ids = batch.get('emotion_ids', None)
        mask_context_input_ids = batch.get('mask_context_input_ids', None)
        mask_context_output_ids = batch.get('mask_context_output_ids', None)
        mask_context_attention_mask = batch.get(
            'mask_context_attention_mask', None)

        return self.model(batch_idx=batch_idx, input_ids=input_ids, attention_mask=attention_mask, img_ids=img_ids, neg_img_ids=neg_img_ids,
                          test=test, cands=cands, emotion_ids=emotion_ids, mask_context_input_ids=mask_context_input_ids, mask_context_output_ids=mask_context_output_ids,
                          mask_context_attention_mask=mask_context_attention_mask)

    def compute_acc(self, sorted_idx, labels, k=5):
        idx = sorted_idx[:, :k]
        labels = labels.unsqueeze(-1).expand_as(idx)
        cor_ts = torch.eq(idx, labels).any(dim=-1)
        cor = cor_ts.sum()
        total = cor_ts.numel()
        return cor, total

    def compute_map(self, sorted_idx, labels):
        l = labels.unsqueeze(-1)
        w = sorted_idx == l
        _, idx = w.nonzero(as_tuple=True)
        s = 1. / (idx + 1)
        return torch.sum(s), labels.size(0)

    def training_step(self, batch, batch_idx):
        # logger.debug(batch)
        outputs = self.run_model_from_batch(batch, batch_idx)

        loss = outputs[0]

        self.log('train_loss', loss)
        # logger.info(f"loss:{loss}")
        return loss

    def training_step_end(self, *args, **kwargs):
        self.model.temperature.data = torch.clamp(
            self.model.temperature.data, 0, np.log(100))
        return super().training_step_end(*args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        # if self.args.val_data_path:
        self.model.prepare_for_test()
        return super().on_validation_epoch_start()

    def log_res(self, batch, sorted_idx, batch_idx, k=5):
        # valtest batch size == 1
        return
        label = batch['img_ids'][0]
        input_ids = batch['input_ids']
        input_text = self.model.bert_tokenizer.decode(
            input_ids.squeeze(0).tolist())
        topk_idx = sorted_idx[0, :k].tolist()
        label_text = self.id2name[label]
        cands = batch.get('cands', None)
        if cands:
            # logger.debug(topk_idx, cands)
            topk_idx = [cands[0][idx] for idx in topk_idx]
            # assert label in topk_idx

        topk_text = [self.id2name[idx] for idx in topk_idx]
        logger.info(
            f"input_text:{input_text}, label:{label_text}, topk_text:{topk_text}, batch_idx:{batch_idx}")

    def validation_step(self, batch, batch_idx):
        outputs = self.run_model_from_batch(batch, batch_idx, test=True)
        # labels = batch['labels']
        if self.args.mode == 'predict_img_label':
            return
        logits = outputs[0]
        labels = outputs[1]
        cands = outputs[2]  # when args.test_with_cand==True
        if cands is not None:
            cands = [int(cand) for cand in cands]
        _, idx = torch.sort(logits, dim=-1, descending=True)
        assert len(labels) == 1
        # print(cands, idx)
        if cands is not None:
            return_preds = torch.tensor(cands)[idx.squeeze(0)].tolist()
        else:
            return_preds = torch.arange(idx.size(1))[idx.squeeze(0)].tolist()
        # print(return_preds)
        return_labels = labels.item()

        self.log_res(batch, idx, batch_idx=batch_idx, k=11)

        if cands:
            # logger.debug(cands, labels)
            for i, cand in enumerate(cands):
                if int(cand) == labels.item():
                    labels[0] = i
                    # logger.debug(f'label after mapping:{i}')
                    break

        topks = [1, 2, 5] if not self.args.test_with_cand else [1, 2, 5]
        # self.valtest_acc5.update(*self.compute_acc(idx, labels, topks[0]))
        # self.valtest_acc30.update(*self.compute_acc(idx, labels, topks[1]))
        # self.valtest_acc90.update(*self.compute_acc(idx, labels, topks[2]))
        # self.valtest_map.update(*self.compute_map(idx, labels))
        cor1, tot1 = self.compute_acc(idx, labels, topks[0])
        cor2, tot2 = self.compute_acc(idx, labels, topks[1])
        cor5, tot5 = self.compute_acc(idx, labels, topks[2])
        cor, tot = self.compute_map(idx, labels)
        self.valtest_acc5.update(cor1, tot1)
        self.valtest_acc30.update(cor2, tot2)
        self.valtest_acc90.update(cor5, tot5)
        self.valtest_map.update(cor, tot)
        return [(cor1 / tot1).item(), (cor2 / tot2).item(), (cor5 / tot5).item(), (cor / tot).item()], return_preds, return_labels

    def on_validation_epoch_end(self) -> None:
        if self.args.mode == 'predict_img_label':
            logger.debug(self.model.predict_id2ocr)
            logger.debug(len(self.model.predict_id2ocr))
            with open('./data/ocr_predict.json', 'w', encoding='utf-8') as f:
                json.dump(self.model.predict_id2ocr, f,
                          indent=2, ensure_ascii=False)
            return
        logger.info(f"\nepoch {self.current_epoch} end, total:{self.valtest_acc5.total}, val acc5:{self.valtest_acc5.compute():.4f},\
            acc30:{self.valtest_acc30.compute():.4f},acc90:{self.valtest_acc90.compute():.4f},map:{self.valtest_map.compute():.4f}")
        self.valtest_acc5.reset()
        self.valtest_acc30.reset()
        self.valtest_acc90.reset()
        self.valtest_map.reset()
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.model.prepare_for_test()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.on_validation_epoch_end()
        print(len(outputs), len(outputs[0]))
        print(outputs[0])

        cols = ['r@1', 'r@2', 'r@5', 'mrr']
        res = {col: [] for col in cols}
        pred_res = []
        for item in outputs:
            metrics, pred, answer = item
            pred_res.append({'pred': pred, 'answer': answer})
            for i, val in enumerate(metrics):
                res[cols[i]].append(val)

        def get_filename_withoutext(path):
            filename_withext = os.path.basename(path)
            name, ext = os.path.splitext(filename_withext)
            return name

        outpath = f'result/{get_filename_withoutext(self.args.test_data_path)}_{self.args.test_with_cand}.json'
        with open(outpath, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        pred_outpath = f'result/{get_filename_withoutext(self.args.test_data_path)}_{self.args.test_with_cand}_pred.json'
        with open(pred_outpath, 'w') as f:
            json.dump(pred_res, f, ensure_ascii=False, indent=2)       

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import LambdaLR
        logger.info(
            f"img lr:{self.args.img_lr}, fix text:{self.args.fix_text}, fix img:{self.args.fix_img}")
        for name, v in self.model.named_parameters():
            print(name)
        assert self.args.model_choice == 'use_img_clip'
        img_params = [var for name, var in self.model.named_parameters(
        ) if name.startswith('img_clip') and var.requires_grad]
        other_params = [var for name, var in self.model.named_parameters() if
                        (not name.startswith('text_clip')) and (not name.startswith('img_clip')) and var.requires_grad]
        logger.info(
            f"len img params:{len(img_params)}, len total params:{len(list(self.model.parameters()))}")
        optimizer = AdamW([{'params': img_params, 'lr': self.args.img_lr}, {
                          'params': other_params}], lr=self.args.other_lr, betas=(0.9, 0.98), weight_decay=0.2)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.2)
        # logger.info(f"Total training steps:{self.num_training_steps}")
        # num_warmup_steps = int(0.01 * self.num_training_steps)
        num_warmup_steps = 0
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, self.num_training_steps)
        # num_warmup_steps = 500
        # def lr_lambda(current_step):
        #     if current_step < num_warmup_steps:
        #         return 1.0
        #     else:
        #         return 10.0
        # scheduler = LambdaLR(optimizer, lr_lambda)
        logger.info("use scheduler!")
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
        # return AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # return torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.9)
        # return AdamW(self.model.parameters(), lr=self.lr)


def test_emotion_prediction(args):
    print('test emotion prediction!')
    model = PLModel(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # print(ckpt.keys())
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda')
    model.eval().to(device)
    pld = PLDataLoader(args, model.model.bert_tokenizer)
    pld.setup('test')
    dataset = pld.test_dataset
    batch_idx = 0
    scores = []

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        if data['emotion_id'] == -100:
            continue

        batch = pld.collate_fn([data])
        # print(batch)
        batch = batch_to_device(batch, device)
        outputs = model.run_model_from_batch(batch, batch_idx, )
        # print(data['emotion_id'], outputs)
        scores.append(float(data['emotion_id'] == outputs.item()))
        batch_idx += 1
        # break

    print(f'valid samples:{len(scores)}, mean accuracy for predicting emotion:{np.mean(scores)}')


def test_sentence_gradient(args):
    print('test sentence gradient!')
    model = PLModel(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # print(ckpt.keys())
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda')
    model.eval().to(device)
    pld = PLDataLoader(args, model.model.bert_tokenizer)
    pld.setup('test')
    dataset = pld.test_dataset
    batch_idx = 0
    scores = []

    tokenizer = model.model.bert_tokenizer

    res = []

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        # if data['emotion_id'] == -100:
        #     continue

        batch = pld.collate_fn([data])
        # print(batch)
        batch = batch_to_device(batch, device)
        outputs = model.run_model_from_batch(batch, batch_idx, ) # grad
        grad, success = outputs
        print(f'success:{success}')
        # if not success:
        #     continue
        # print(f'grad:{outputs.size()}')
        input_ids = batch['input_ids'].squeeze(0).tolist()
        # print(input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        d = {}
        d['tokens'] = ''.join(tokens)
        grad_norm = torch.norm(grad.squeeze(0), 2, dim=-1)
        d['token_scores'] = []
        # print(grad_norm.size())
        for j, token in enumerate(tokens):
            print(token, grad_norm[j].item())
            d['token_scores'].append([token, grad_norm[j].item()])
        res.append(d)
        # print(data['emotion_id'], outputs)
        # scores.append(float(data['emotion_id'] == outputs.item()))
        batch_idx += 1
    
        # break
    with open('./log_gradient_test_easy.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f'valid samples:{len(scores)}, mean accuracy for predicting emotion:{np.mean(scores)}')

@dataclass
class Arguments:
    train_data_path: Optional[str] = field(default='./data/train_pair.json')
    val_data_path: Optional[str] = field(default='./data/validation_pair.json')
    test_data_path: Optional[str] = field(
        default='./data/validation_pair.json')
    gen_data_path: Optional[str] = field(default='./data/validation_gen.json')
    gen_out_data_path: Optional[str] = field(
        default='./data/validation_gen_out.json')
    img_lr: Optional[float] = field(default=5e-7)
    other_lr: Optional[float] = field(default=1e-5)
    gpus: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=2021)
    epochs: Optional[int] = field(default=10)
    train_batch_size: Optional[int] = field(default=8)
    valtest_batch_size: Optional[int] = field(default=1)
    gen_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    mode: Optional[str] = field(default='train')
    ckpt_path: Optional[str] = field(
        default='')
    ckpt_epoch: Optional[int] = field(default=-1)

    pretrain_path: Optional[str] = field(
        default='./ckpt/chinese-roberta-wwm-ext')
    bert_pretrain_path: Optional[str] = field(
        default='bert-base-chinese')
    pl_root_dir: Optional[str] = field(default='logs/clip')
    id2img_path: Optional[str] = field(default='./data/id2img.json')
    img_dir: Optional[str] = field(default='./data/meme_set')
    fix_text: Optional[bool] = field(default=False)
    fix_img: Optional[bool] = field(default=False)
    model_choice: Optional[str] = field(default='use_img_clip')
    max_image_id: Optional[int] = field(default=307)
    max_emotion_id: Optional[int] = field(default=52)
    init_temp: Optional[float] = field(default=np.log(1/0.07))
    sent_num: Optional[int] = field(default=1)
    num_workers: Optional[int] = field(default=32)
    test_with_cand: Optional[bool] = field(default=False)
    add_emotion_task: Optional[bool] = field(default=False)
    add_predict_context_task: Optional[bool] = field(default=False)
    add_predict_img_label_task: Optional[bool] = field(default=False)
    add_ocr_info: Optional[bool] = field(default=False)
    max_img_label_mask_num: Optional[int] = field(default=11, metadata={
                                                  'help': 'img label length plus one position for cls(start) token'})
    ocr_path: Optional[str] = field(default='./data/ocr_max10.json')
    emotion_weight: Optional[float] = field(default=0.1)
    ctx_weight: Optional[float] = field(default=0.05)
    label_weight: Optional[float] = field(default=0.1)

    def __post_init__(self):
        # if self.add_predict_img_label_task is True:
        #     assert self.add_ocr_info is True
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            names = os.listdir(self.ckpt_path)
            if len(names) == 1:
                self.ckpt_path = os.path.join(self.ckpt_path, names[0])
            elif len(names) > 1:
                if(self.ckpt_epoch == -1):
                    raise Exception(
                        "More than 1 checkpoint but ckpt_epoch is -1")
                for name in names:
                    if f'epoch={self.ckpt_epoch}-' in name:
                        self.ckpt_path = os.path.join(self.ckpt_path, name)
                        break
            else:
                raise Exception(f'No checkpoints in {self.ckpt_path}')

            logger.info(f'real ckpt_path:{self.ckpt_path}')


def main(args):
    pl.seed_everything(args.seed)
    if args.mode == 'train' or args.mode == 'pretrain':
        model = PLModel(args)
        if args.ckpt_path:
            logger.info(f'load from {args.ckpt_path}')
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['state_dict'])
        if args.model_choice == 'use_clip_repo':
            pld = PLDataLoader(args)
        else:
            pld = PLDataLoader(args, model.model.bert_tokenizer)
        checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                              verbose=True)
        # earlystop_callback = EarlyStopping(monitor='val_loss',
        #                                    verbose=True,
        #                                    mode='min')
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            # num_sanity_val_steps=0,
            # limit_train_batches=16,
            # limit_val_batches=0.01,
            # gradient_clip_val=1.0,
            max_epochs=args.epochs,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            default_root_dir=args.pl_root_dir)
        trainer.fit(model, pld)
    elif args.mode == 'test' or args.mode == 'gen':
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'])
        pld = PLDataLoader(args, model.model.bert_tokenizer)
        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=args.epochs,
                             default_root_dir=args.pl_root_dir)
        trainer.test(model, datamodule=pld)

    elif args.mode == 'predict_img_label':
        model = PLModel(args)
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # print(ckpt.keys())
        model.load_state_dict(ckpt['state_dict'], strict=False)
        pld = PLDataLoader(args, model.model.bert_tokenizer)

        trainer = pl.Trainer(gpus=args.gpus,
                             max_epochs=args.epochs,
                             default_root_dir=args.pl_root_dir)
        trainer.test(model, datamodule=pld)

    elif args.mode == 'predict_emotion':
        test_emotion_prediction(args)

    elif args.mode == 'test_gradient':
        test_sentence_gradient(args)


if __name__ == '__main__':
    try_create_dir('./logs')
    parser = HfArgumentParser(Arguments)
    args, = parser.parse_args_into_dataclasses()
    main(args)
