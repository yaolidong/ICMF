import torch
import json
import numpy as np
import os
import os.path as osp
from collections import Counter
import pickle

from .utils import get_adjr

class EADataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Collator_base(object):
    def __call__(self, batch):
        return np.array(batch)


def split_alignment_pairs(ills, data_rate):

    pairs = np.asarray(ills, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("reference alignments must have shape [N, 2]")
    train_rate = float(data_rate)
    if not 0.0 < train_rate < 1.0:
        raise ValueError(f"data_rate must be in (0, 1), got {train_rate}")

    train_count = int(len(pairs) * train_rate)
    if train_count <= 0:
        raise ValueError("training split is empty")
    train_pairs = pairs[:train_count]
    non_training_pairs = pairs[train_count:]
    return train_pairs, non_training_pairs


def seeded_alignment_split(ills, data_rate, random_seed):

    pairs = np.asarray(ills, dtype=np.int32)


    generator = np.random.RandomState(int(random_seed))
    shuffled = pairs[generator.permutation(len(pairs))]
    train, evaluation = split_alignment_pairs(
        shuffled, data_rate=data_rate
    )
    if not np.array_equal(np.vstack((train, evaluation)), shuffled):
        raise AssertionError("alignment split changed the shuffled reference list")
    return train, evaluation


def _alignment_partition_record(values):
    if values is None:
        return None
    array = np.ascontiguousarray(values, dtype="<i4")
    return {
        "count": int(array.shape[0]),
    }


def _alignment_entity_overlap(first, second):
    if first is None or second is None:
        return None
    return {
        "left_entities": len(set(first[:, 0]).intersection(second[:, 0])),
        "right_entities": len(set(first[:, 1]).intersection(second[:, 1])),
    }


def alignment_split_manifest(train, validation, random_seed, data_rate):

    return {
        "partition_unit": "alignment_pair",
        "random_seed": int(random_seed),
        "data_rate": float(data_rate),
        "validation_rate": 0.0,
        "train": _alignment_partition_record(train),
        "validation": _alignment_partition_record(validation),
        "test": None,
        "entity_overlap_audit": {
            "train_validation": _alignment_entity_overlap(train, validation),
            "train_test": None,
            "validation_test": None,
        },
    }


def permute_available_features_within_kgs(
    features,
    availability,
    left_entities,
    right_entities,
    random_seed,
):

    values = np.asarray(features)
    mask = np.asarray(availability, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("modality features must be a rank-2 array")
    if mask.shape != (values.shape[0],):
        raise ValueError("availability mask must have one entry per feature row")

    output = values.copy()
    seed_sequence = np.random.SeedSequence(int(random_seed))
    side_seeds = seed_sequence.spawn(2)
    manifest = {
        "type": "within_kg_available_row_derangement",
        "random_seed": int(random_seed),
        "sides": {},
    }

    for side_name, entity_ids, side_seed in zip(
        ("left", "right"),
        (left_entities, right_entities),
        side_seeds,
    ):
        ids = np.asarray(entity_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError(f"{side_name} entity ids must be one-dimensional")
        if ids.size and (ids.min() < 0 or ids.max() >= values.shape[0]):
            raise ValueError(f"{side_name} entity ids are outside the feature matrix")
        available_ids = ids[mask[ids]]
        donor_ids = available_ids.copy()
        if available_ids.size > 1:
            generator = np.random.default_rng(side_seed)
            recipients = generator.permutation(available_ids)
            donors = np.roll(recipients, 1)
            output[recipients] = values[donors]
            donor_by_recipient = dict(zip(recipients.tolist(), donors.tolist()))
            donor_ids = np.asarray(
                [donor_by_recipient[int(entity_id)] for entity_id in available_ids],
                dtype=np.int64,
            )

        manifest["sides"][side_name] = {
            "entity_count": int(ids.size),
            "available_count": int(available_ids.size),
            "moved_count": int(np.count_nonzero(available_ids != donor_ids)),
        }

    if not np.array_equal(
        np.abs(output).sum(axis=1) > 1e-6,
        mask,
    ):
        raise AssertionError("relation permutation changed the availability mask")
    return output, manifest


def load_data(logger,args):
    assert args.data_choice in ["DBP15K", "FBYG15K", "FBDB15K"]
    return load_eva_data(logger,args)

def load_eva_data(logger,args):

    file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
    real_choice = args.data_choice



    if not osp.exists(file_dir) and args.data_choice in ["FBDB15K", "FBYG15K"]:
        mapping = {"FBDB15K": "FB15K_DB15K", "FBYG15K": "FB15K_YAGO15K"}
        alt_choice = mapping.get(args.data_choice, args.data_choice)
        alt_dir = osp.join(args.data_path, alt_choice)
        if osp.exists(alt_dir):
            file_dir = alt_dir
            real_choice = alt_choice
    lang_list = [1,2]
    ent2id_dict,ills,triples,r_hs,_r_ts,_ids = read_raw_data(file_dir,lang_list)
    e1 = os.path.join(file_dir,'ent_ids_1')
    e2 = os.path.join(file_dir,'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    if "FB" in file_dir:

        cand_paths = [
            osp.join(args.data_path, "pkls", f"{args.data_choice}_id_img_feature_dict.pkl"),
            osp.join(args.data_path, "pkls", f"{real_choice}_id_img_feature_dict.pkl"),
            osp.join(args.data_path, f"{args.data_choice}_id_img_feature_dict.pkl"),
            osp.join(args.data_path, f"{real_choice}_id_img_feature_dict.pkl"),
            osp.join(file_dir, f"{args.data_choice}_id_img_feature_dict.pkl"),
            osp.join(file_dir, f"{real_choice}_id_img_feature_dict.pkl"),
        ]
        img_vec_path = None
        for pp in cand_paths:
            if osp.exists(pp):
                img_vec_path = pp
                break
        if img_vec_path is None:
            raise FileNotFoundError(f"cannot find image feature pkl for {args.data_choice}: tried {cand_paths}")
    else:

        img_vec_path = osp.join(args.data_path, "pkls", args.data_split + "_GA_id_img_feature_dict.pkl")

    assert img_vec_path is not None and osp.exists(img_vec_path)
    img_features, img_mask = load_img(logger,ENT_NUM,img_vec_path,triples=triples)
    logger.info(f"image feature shape:{img_features.shape}")

    train_ill, eval_ill = seeded_alignment_split(
        ills,
        args.data_rate,
        args.random_seed,
    )
    split_manifest = alignment_split_manifest(
        train_ill,
        eval_ill,
        args.random_seed,
        args.data_rate,
    )
    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))
    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    logger.info(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
    logger.info(
        f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    rel_features = load_relation(ENT_NUM,triples,1000)
    logger.info(f"relation feature shape:{rel_features.shape}")
    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)
    logger.info(f"attribute feature shape:{att_features.shape}")


    rel_mask = (np.abs(rel_features).sum(axis=1) > 1e-6).astype(np.bool_)
    attr_mask = (np.abs(att_features).sum(axis=1) > 1e-6).astype(np.bool_)
    rel_features_permuted = None
    relation_warmup_epochs = int(
        getattr(args, "relation_permuted_warmup_epochs", 0)
    )
    relation_intervention_manifest = {
        "type": str(getattr(args, "relation_intervention", "clean")),
    }
    if str(getattr(args, "relation_intervention", "clean")) == "permuted":
        rel_features, relation_intervention_manifest = permute_available_features_within_kgs(
            rel_features,
            rel_mask,
            left_ents,
            right_ents,
            getattr(args, "relation_permutation_seed", 20260722),
        )
        logger.info(
            "relation intervention manifest: %s",
            json.dumps(relation_intervention_manifest, sort_keys=True),
        )
    elif relation_warmup_epochs > 0:
        rel_features_permuted, permutation_manifest = (
            permute_available_features_within_kgs(
                rel_features,
                rel_mask,
                left_ents,
                right_ents,
                getattr(args, "relation_permutation_seed", 20260722),
            )
        )
        relation_intervention_manifest = {
            "type": "training_warmup_relation_derangement",
            "scope": "training_only",
            "warmup_epochs": relation_warmup_epochs,
            "evaluation_input": "clean",
            "permutation": permutation_manifest,
        }
        logger.info(
            "relation intervention manifest: %s",
            json.dumps(relation_intervention_manifest, sort_keys=True),
        )

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(
        f"train ill num:\t {train_ill.shape[0]} \t "
        f"held-out eval ill num:\t {eval_ill.shape[0]} \t "
        "separate test ill num:\t 0 (legacy shared held-out protocol)"
    )
    logger.info("-------------------------")
    logger.info(
        "alignment split manifest: %s",
        json.dumps(split_manifest, sort_keys=True),
    )

    input_idx = torch.LongTensor(np.arange(ENT_NUM))
    adj = get_adjr(ENT_NUM, triples, norm=True)
    train_ill = EADataset(train_ill)
    eval_ill = EADataset(eval_ill)

    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,

        'left_ents': np.asarray(left_ents, dtype=np.int32),
        'right_ents': np.asarray(right_ents, dtype=np.int32),
        'images_list': img_features,
        'rel_features': rel_features,
        'rel_features_permuted': rel_features_permuted,
        'att_features': att_features,
        'img_mask': img_mask,
        'rel_mask': rel_mask,
        'attr_mask': attr_mask,
        'alignment_split_manifest': split_manifest,
        'relation_intervention_manifest': relation_intervention_manifest,
        'input_idx': input_idx,
        'adj': adj,
    }, {"left": left_non_train, "right": right_non_train}, train_ill, eval_ill

def read_raw_data(file_dir, lang=[1, 2]):

    print('loading raw data...')

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids

    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in lang])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids



def load_attr(fns, e, ent2id, topA=1000):
    if topA <= 0:
        raise ValueError("topA must be positive")
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    selected_count = min(topA, len(fre))
    for i in range(selected_count):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, selected_count), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0

    return attr


def load_relation(e, KG, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, _) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_img(logger, e_num, path, triples=None):
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)

    img_mask = np.zeros(e_num, dtype=np.bool_)
    for ent_id in img_dict.keys():
        if ent_id < e_num:
            img_mask[ent_id] = True


    from collections import defaultdict
    neighbor_dict = defaultdict(list)
    for triple in triples:
        h, _r, t = triple[0], triple[1], triple[2]
        if t in img_dict:
            neighbor_dict[h].append(t)
        if h in img_dict:
            neighbor_dict[t].append(h)

    all_img_emb_normal = np.random.normal(mean, std, mean.shape[0])
    img_embd = []
    n_neighbor_fill = 0
    n_random_fill = 0
    for i in range(e_num):
        if i in img_dict:
            img_embd.append(img_dict[i])
        elif len(neighbor_dict[i]) > 0:

            neighbor_imgs = np.array([img_dict[nid] for nid in neighbor_dict[i]])
            img_embd.append(np.mean(neighbor_imgs, axis=0))
            n_neighbor_fill += 1
        else:
            img_embd.append(all_img_emb_normal)
            n_random_fill += 1
    img_embd = np.array(img_embd)

    n_has_img = len(img_dict)
    logger.info(
        f"{(100 * n_has_img / e_num):.2f}% entities have images, "
        f"neighbor_fill={n_neighbor_fill}, random_fill={n_random_fill}"
    )
    return img_embd, img_mask
