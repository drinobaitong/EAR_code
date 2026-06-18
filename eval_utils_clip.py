from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import torch
import numpy as np
import json
from collections import OrderedDict
from tqdm import tqdm
from os.path import dirname, abspath
import math
import itertools

ear_dir = dirname(abspath(__file__))
sys.path.insert(0, ear_dir)
sys.path.insert(0, os.path.join(ear_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(ear_dir, 'densevid_eval3/SODA'))

from densevid_eval3.eval_soda import eval_soda
from densevid_eval3.eval_para import eval_para
from densevid_eval3.eval_dvc import eval_dvc
from densevid_eval3.eval_tal import eval_tal

from misc.plot_proposal_distribution import main as plot_proposal_distribution


def get_iou(pred_ts, gt_ts):

    pred_start, pred_end = float(pred_ts[0]), float(pred_ts[1])
    gt_start, gt_end = float(gt_ts[0]), float(gt_ts[1])

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0.0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    if union == 0:
        return 0.0

    return inter / union


def get_miou(predictions, groundtruths):

    gt_by_video = {}
    for unique_id, gt_info in groundtruths.items():
        video_id = gt_info['video_id'].lower().strip()
        if not video_id:
            continue
        if video_id not in gt_by_video:
            gt_by_video[video_id] = {'timestamps': []}
        gt_by_video[video_id]['timestamps'].append(gt_info['timestamp'])

    pred_by_video = {}
    for unique_id, pred_list in predictions.items():
        if '-' not in unique_id:
            continue
        video_id = unique_id.split('-')[0].lower().strip()
        if not video_id:
            continue
        if video_id not in pred_by_video:
            pred_by_video[video_id] = []
        pred_by_video[video_id].extend(pred_list)

    common_ids = set(gt_by_video.keys()) & set(pred_by_video.keys())
    print(f"\n[GroundingDebug] matched video ids: {len(common_ids)}")
    print(f"[GroundingDebug] matched video id examples: {list(common_ids)[:5]}")

    all_max_ious = []
    missing_num = 0
    all_num = len(gt_by_video)
    matched_num = 0

    for video_id in gt_by_video.keys():
        if video_id not in pred_by_video:
            missing_num += 1
            continue
        matched_num += 1
        gt_segments = gt_by_video[video_id]['timestamps']
        pred_segments = [p['timestamp'] for p in pred_by_video[video_id]]

        for pred in pred_segments:
            max_iou = max([get_iou(pred, gt) for gt in gt_segments])
            all_max_ious.append(max_iou)

    miou = sum(all_max_ious) / (len(all_max_ious) + 1e-8) if len(all_max_ious) > 0 else 0.0
    print(
        f'Calculating mIOU: total videos: {all_num}, missing videos: {missing_num}, matched videos: {matched_num}, total matched segments: {len(all_max_ious)}')
    return miou


def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5):

    gt_by_video = {}
    for unique_id, gt_info in groundtruths.items():
        video_id = gt_info['video_id'].lower().strip()
        if not video_id:
            continue
        if video_id not in gt_by_video:
            gt_by_video[video_id] = {'timestamps': []}
        gt_by_video[video_id]['timestamps'].append(gt_info['timestamp'])

    pred_by_video = {}
    for unique_id, pred_list in predictions.items():
        if '-' not in unique_id:
            continue
        video_id = unique_id.split('-')[0].lower().strip()
        if not video_id:
            continue
        if video_id not in pred_by_video:
            pred_by_video[video_id] = []
        pred_by_video[video_id].extend(pred_list)

    hit = np.zeros(shape=(len(gt_by_video.keys()),), dtype=np.float32)
    all_num = len(gt_by_video)
    missing_num = 0
    matched_num = 0

    for idd, video_id in enumerate(gt_by_video.keys()):
        if video_id not in pred_by_video:
            missing_num += 1
            continue
        matched_num += 1
        pred_list = sorted(pred_by_video[video_id], key=lambda x: x['score'], reverse=True)[:max_proposal_num]
        pred_segments = [p['timestamp'] for p in pred_list]
        gt_segments = gt_by_video[video_id]['timestamps']

        has_hit = False
        for pred in pred_segments:
            if any([get_iou(pred, gt) >= iou_threshold for gt in gt_segments]):
                has_hit = True
                break
        if has_hit:
            hit[idd] = 1.0

    avg_recall = np.sum(hit) / (len(hit) + 1e-8) if len(hit) > 0 else 0.0
    print(
        f'Calculating Recall@{max_proposal_num}: total videos: {all_num}, missing videos: {missing_num}, matched videos: {matched_num}')
    return avg_recall

def eval_grounding(result_file, gt_file):

    results = json.load(open(result_file, 'r'))['results']
    groundtruth_data = json.load(open(gt_file, 'r'))

    out_grounding_data = {}
    for video_id in groundtruth_data.keys():
        gd = groundtruth_data[video_id]
        for anno_id in range(len(gd['timestamps'])):
            unique_anno_id = video_id + '-' + str(anno_id)
            out_grounding_data[unique_anno_id] = {
                'video_id': video_id,
                'anno_id': anno_id,
                'timestamp': gd['timestamps'][anno_id]
            }
    groundtruth_data = out_grounding_data

    scores = {}
    scores['mIOU'] = get_miou(results, groundtruth_data)

    for iou, max_proposal_num in list(itertools.product([0.7, 0.5, 0.3, 0.1], [1, 5])):
        recall_key = f'R@{max_proposal_num}IOU{iou}'
        scores[recall_key] = get_recall_at_k(results, groundtruth_data, iou, max_proposal_num)

    return scores


def calculate_avg_proposal_num(json_path):
    data = json.load(open(json_path))
    return np.array([len(v) for v in data['results'].values()]).mean()


def convert_tapjson_to_dvcjson(tap_json, dvc_json):
    data = json.load(open(tap_json, 'r'))
    data['version'] = "VERSION 1.0"
    data['external_data'] = {'used:': True, 'details': "C3D pretrained on Sports-1M"}

    all_names = list(data['results'].keys())
    for video_name in all_names:
        for p_info in data['results'][video_name]:
            p_info['timestamp'] = p_info.pop('segment')
            p_info['proposal_score'] = p_info.pop('score')
            p_info['sentence_score'] = p_info.pop('sentence_score', 0)
        data['results']["v_" + video_name] = data['results'].pop(video_name)
    json.dump(data, open(dvc_json, 'w'))


def convert_dvcjson_to_tapjson(dvc_json, tap_json):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        event_num = len(data[video_name])
        timestamps = [data[video_name][i]['timestamp'] for i in range(event_num)]
        sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            score = data[video_name][i].get('proposal_score', 1.0)
            video_info.append({'segment': timestamp, 'score': score, 'sentence': sentences[i],
                               'sentence_score': data[video_name][i].get('sentence_score', 0)})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))


def convert_gtjson_to_tapjson(gt_json, tap_json):
    data = json.load(open(gt_json, 'r'))
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        timestamps = data[video_name]['timestamps']
        sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1., 'sentence': sentences[i]})
        out['results'][video_name[2:]] = video_info
    with open(tap_json, 'w') as f:
        json.dump(out, f)


def get_topn_from_dvcjson(dvc_json, out_json, top_n=3, ranking_key='proposal_score', score_thres=-1e8):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}
    all_names = list(data.keys())
    num = 0
    bad_vid = 0
    for video_name in all_names:
        info = data[video_name]
        new_info = sorted(info, key=lambda x: x[ranking_key], reverse=True)
        new_info = [p for p in new_info if p[ranking_key] > score_thres]
        new_info = new_info[:top_n]
        out['results'][video_name] = new_info
        num += len(new_info)
        if len(new_info) == 0:
            bad_vid += 1
            out['results'].pop(video_name)
    print('average proosal number: {}'.format(num / len(all_names)))
    print('bad videos number: {}'.format(bad_vid))
    print('good videos number: {}'.format(len(out['results'])))
    with open(out_json, 'w') as f:
        json.dump(out, f)


def eval_metrics_grounding(g_filename, gt_filename):
    score = collections.defaultdict(lambda: -1)
    grounding_scores = eval_grounding(g_filename, gt_filename)
    for key in grounding_scores.keys():
        score[key] = grounding_scores[key]
    return score


def eval_metrics(dvc_filename, gt_filenames, para_gt_filenames, alpha=0.3, temperature=2.0, cl_score_weight=0.,
                 ranking_key='proposal_score', rerank=False, dvc_eval_version='2018', infer_mode=False):
    score = collections.defaultdict(lambda: -1)
    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames, version=dvc_eval_version,
                         infer_mode=infer_mode)
    dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    dvc_score.update({'MetaScore': dvc_score['METEOR'] + dvc_score['soda_c']})
    score.update(dvc_score)
    return score


def save_dvc_json(out_json, path):
    with open(path, 'w') as f:
        if 'results' in out_json:
            out_json['valid_video_num'] = int(len(out_json['results']))
            avg_proposal = np.array([len(v) for v in out_json['results'].values()]).mean()

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        out_json = convert_numpy(out_json)
        json.dump(out_json, f)


def reranking(p_src, alpha, cl_score_weight, temperature, fix_topN=-1, increase_num=0):
    print('alpha: {}, temp: {}'.format(alpha, temperature))
    d = json.load(open(p_src))
    d_items = list(d['results'].items())
    for k, v in d_items:
        if True:
            sent_scores = [p['sentence_score'] / (float(len(p['sentence'].split())) ** (temperature) + 1e-5) for p in v]
            prop_score = [p['proposal_score'] for p in v]
            cl_score = [p['cl_score'] for p in v]
            joint_score = alpha * (np.array(sent_scores)) + (np.array(prop_score))
        for i, p in enumerate(v):
            p['joint_score'] = joint_score[i]
        v = sorted(v, key=lambda x: x['joint_score'], reverse=True)
        topN = v[0]['pred_event_count'] if fix_topN < 0 else fix_topN
        r = increase_num - math.floor(increase_num)
        if r > 0:
            increase_num_ = math.floor(increase_num) + np.random.binomial(1, p=r, size=None)
        else:
            increase_num_ = int(increase_num)
        topN += increase_num_
        v = v[:int(topN)]
        v = sorted(v, key=lambda x: x['timestamp'])
        d['results'][k] = v
    save_path = p_src + '_rerank_alpha{}_temp{}.json'.format(alpha, temperature)
    save_dvc_json(d, save_path)
    return save_path


def evaluate(model, memory_bank, criterion, contrastive_criterion, postprocessors, loader, dvc_json_path, logger=None,
             score_threshold=0,
             alpha=0.3, dvc_eval_version='2018', device='cuda', debug=False, skip_lang_eval=False, infer_mode=False,
             sent_embedder=None, tokenizer=None):
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': None}}

    out_json_g = {'results': {}}
    aux_out_json_g = {'results': {}}

    opt = loader.dataset.opt

    loss_sum = OrderedDict()
    len_check = 0
    model_core = model.module if hasattr(model, 'module') else model
    if hasattr(model_core, 'reset_anchor_analysis_stats'):
        model_core.reset_anchor_analysis_stats()
    with torch.set_grad_enabled(False):
        for dt in tqdm(loader, disable=opt.disable_tqdm):
            len_check += 1
            dt = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}

            dt['video_target'] = [
                {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]
            gt_bank = None
            if opt.ideal_test:
                gt_bank = load_val_gt(opt)
                gt_bank['vid_sent_embeds'] = torch.tensor(gt_bank['vid_sent_embeds']).to('cuda')

            captions = list()
            for video_sents in dt['cap_raw']:
                captions.extend(video_sents)

            output, loss = model(dt, criterion, contrastive_criterion, opt.transformer_input_type, memory_bank,
                                 eval_mode=True, sent_embedder=sent_embedder, gt_bank=gt_bank)
            orig_target_sizes = dt['video_length'][:, 1]

            weight_dict = criterion.weight_dict
            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)

            for loss_k, loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0) + loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            results = postprocessors['bbox'](output, orig_target_sizes, loader, model, tokenizer)

            results_g, cl_scores = postprocessors['bbox'].forward_grounding(output, orig_target_sizes,
                                                                            dt['video_target'])
            aux_results_g, aux_cl_scores = postprocessors['bbox'].forward_grounding(output['aux_outputs'][-1],
                                                                                    orig_target_sizes,
                                                                                    dt['video_target'])
            batch_json = {}
            batch_json_g = {}
            aux_batch_json_g = {}
            for idx, video_name in enumerate(dt['video_key']):
                segment = results[idx]['boxes'].cpu().numpy()
                is_gt_proposals = opt.transformer_input_type == 'gt_proposals'
                segment_num = len(segment)
                raw_boxes = results[idx]['raw_boxes'].cpu().numpy()
                raw_boxes_mask = raw_boxes.sum(1) != 0

                batch_json[video_name] = [
                    {
                        "timestamp": segment[pid].tolist(),
                        "raw_box": raw_boxes[pid].tolist(),
                        "label": results[idx]['labels'][pid].item(),

                        "proposal_score": results[idx]['scores'][pid].item(),
                        "sentence": results[idx]['captions'][pid],
                        "sentence_score": results[idx]['caption_scores'][pid],
                        "cl_score": results[idx]['cl_scores'][pid],
                        'query_id': results[idx]['query_id'][pid].item(),
                        'vid_duration': results[idx]['vid_duration'].item(),
                        'pred_event_count': results[idx]['pred_seq_len'].item(),
                    }
                    for pid in range(segment_num) if
                    results[idx]['scores'][pid].item() > score_threshold and raw_boxes_mask[pid]]
                if results_g:
                    collect_grounding_result(idx, video_name, opt, dt, batch_json_g, results_g)
                if aux_results_g:
                    collect_grounding_result(idx, video_name, opt, dt, aux_batch_json_g, aux_results_g)

            out_json['results'].update(batch_json)
            out_json_g['results'].update(batch_json_g)
            aux_out_json_g['results'].update(aux_batch_json_g)
            if debug and len(out_json['results']) > 5:
                break

    if opt.only_ft_class_head:
        tal_result_json_path = dvc_json_path[:-5] + '.tal.json'
        out_json_tal = collect_tal_result(out_json, loader.dataset.name_map)
        save_dvc_json(out_json_tal, tal_result_json_path)

    save_dvc_json(out_json, dvc_json_path)

    try:
        plot_proposal_distribution(dvc_json_path)
    except:
        pass

    for k in loss_sum.keys():
        loss_sum[k] = np.round(loss_sum[k] / (len(loader) + 1e-5), 3).item()

    if logger is not None:
        logger.info('loss: {}'.format(loss_sum))

    if opt.count_loss_coef > 0:
        dvc_json_path = reranking(dvc_json_path, alpha=alpha, cl_score_weight=opt.eval_matching_score_weight,
                                  temperature=2.0)
    save_dvc_json(out_json_g, dvc_json_path + '.grounding.json')
    save_dvc_json(aux_out_json_g, dvc_json_path + '_aux.grounding.json')
    skip_lang_eval = skip_lang_eval or vars(opt).get('eval_disable_captioning', False)
    if not skip_lang_eval:
        scores = eval_metrics(dvc_json_path,
                              gt_filenames=opt.gt_file_for_eval,
                              para_gt_filenames=opt.gt_file_for_para_eval,
                              alpha=alpha,
                              cl_score_weight=opt.eval_matching_score_weight,
                              rerank=(opt.count_loss_coef > 0),
                              dvc_eval_version=dvc_eval_version,
                              infer_mode=infer_mode
                              )
    else:
        scores = {}
    out_json.update(scores)
    scores_g = eval_metrics_grounding(dvc_json_path + '.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)
    aux_scores_g = eval_metrics_grounding(dvc_json_path + '_aux.grounding.json',
                                          gt_filename=opt.eval_gt_file_for_grounding)
    rename_aux_scores_g = {'aux_' + key: value for key, value in aux_scores_g.items()}
    out_json_g.update(scores_g)
    aux_out_json_g.update(aux_scores_g)
    scores.update(scores_g)
    scores.update(rename_aux_scores_g)

    if hasattr(model_core, 'get_anchor_analysis_stats'):
        anchor_stats = model_core.get_anchor_analysis_stats()
        if getattr(opt, 'enable_anchor_analysis_store', False) and getattr(opt, 'anchor_analysis_path', ''):
            os.makedirs(os.path.dirname(opt.anchor_analysis_path) or '.', exist_ok=True)
            with open(opt.anchor_analysis_path, 'w') as f:
                json.dump(anchor_stats, f, indent=2)
            if logger is not None:
                logger.info('saved anchor analysis stats to {}'.format(opt.anchor_analysis_path))
        for key, value in anchor_stats.items():
            scores[f'anchor_stat_{key}'] = value

    if hasattr(model_core, 'get_atem_analysis_stats'):
        atem_stats = model_core.get_atem_analysis_stats()
        if getattr(opt, 'enable_atem_analysis_store', False) and getattr(opt, 'atem_analysis_path', ''):
            os.makedirs(os.path.dirname(opt.atem_analysis_path) or '.', exist_ok=True)
            with open(opt.atem_analysis_path, 'w') as f:
                json.dump(atem_stats, f, indent=2)
            if logger is not None:
                logger.info('saved ATEM analysis stats to {}'.format(opt.atem_analysis_path))

    if opt.only_ft_class_head:
        score_tal = eval_tal(ground_truth_filename=opt.tal_gt_file, prediction_filename=tal_result_json_path)
        out_json_tal.update(score_tal)
        save_dvc_json(out_json_tal, tal_result_json_path)
        scores.update(score_tal)
    save_dvc_json(out_json, dvc_json_path)
    save_dvc_json(out_json_g, dvc_json_path + '.grounding.json')
    save_dvc_json(aux_out_json_g, dvc_json_path + '_aux.grounding.json')
    return scores, loss_sum


def collect_tal_result(out, name_map):
    tal_out = {'results': {}, 'version': 'VERSION 1.3', 'external_data': {}}
    for key, items in out['results'].items():
        key = key[2:]
        tal_items = []
        for pred in items:
            label = pred['label']
            segment = pred['timestamp']
            score = pred['proposal_score']
            tal_item = {
                'label': name_map.convert_idx2name(label),
                'segment': segment,
                'score': score
            }
            tal_items.append(tal_item)
        tal_out['results'].update({key: tal_items})
    return tal_out


def collect_grounding_result(idx, video_name, opt, dt, batch_json_g, results_g):
    v_name = video_name[2:] if video_name.startswith('v_') else video_name

    for pid in range(len(results_g[idx]['boxes'])):
        unique_key = v_name + '-' + str(pid)

        timestamp = results_g[idx]['boxes'][pid]
        if isinstance(timestamp, np.ndarray):
            timestamp = timestamp.tolist()
        score = float(results_g[idx]['confs'][pid])
        cl_score = float(results_g[idx]['cl_scores'][pid]) if 'cl_scores' in results_g[idx] else 0.0

        batch_json_g[unique_key] = [{
            "timestamp": timestamp,
            "score": score,
            "cl_score": cl_score,
            "sentence": dt['cap_raw'][idx][pid]
        }]


def load_val_gt(args):
    first_iter = True
    for bank_type in args.bank_type:
        rag = os.path.join(args.bank_path, bank_type)  # which domain will be used
        rag = os.path.join(rag, args.visual_feature_type[0])  # matching for backbone

        if args.memory_type == 'clip':
            text_embed = np.load(os.path.join(rag, "gt_embeds.npy"))
            scene_videoID = np.load(os.path.join(rag, "gt_scene_videoID.npy"))
        if first_iter:
            scene_videoIDs = scene_videoID
            text_embeds = text_embed
        else:
            text_embeds = np.concatenate((text_embeds, text_embed))
            scene_videoIDs = np.concatenate((scene_videoIDs, scene_videoID))
        first_iter = False

    gt_bank = {}
    if len(text_embeds) == len(scene_videoIDs):
        gt_bank = {
            "vid_sent_embeds": text_embeds,
            "video_id": scene_videoIDs
        }
    else:
        return -1
    return gt_bank
