import os

from torch.utils.data import DataLoader

from curvspec.corpus.datasets import (
    Dataset4PRVR,
    TxtDataSet4PRVR,
    VisDataSet4PRVR,
    collate_frame_val,
    collate_text_val,
    collate_train,
    read_video_ids,
)
from curvspec.corpus.feature_store import FeatureStore, read_mapping


def _collection_paths(config, split):
    root = config["data_root"]
    collection = config["collection"]
    text_directory = os.path.join(root, collection, "TextData")
    feature_directory = os.path.join(root, collection, "FeatureData", config["visual_feature"])
    caption_path = os.path.join(text_directory, f"{collection}{split}.caption.txt")
    text_feature_path = os.path.join(text_directory, f"roberta_{collection}_query_feat.hdf5")
    mapping_path = os.path.join(feature_directory, "video2frames.txt")
    return caption_path, text_feature_path, feature_directory, mapping_path


def _loader(dataset, config, batch_size, collate_fn, shuffle=False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=config["pin_memory"],
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )


def _build_split(config, split):
    caption_path, text_feature_path, feature_directory, mapping_path = _collection_paths(config, split)
    visual_features = FeatureStore(feature_directory)
    config["visual_feat_dim"] = visual_features.ndims
    video_to_frames = read_mapping(mapping_path)
    video_ids = read_video_ids(caption_path)
    video_dataset = VisDataSet4PRVR(visual_features, video_to_frames, config, video_ids=video_ids)
    text_dataset = TxtDataSet4PRVR(caption_path, text_feature_path, config)
    context_loader = _loader(video_dataset, config, config["eval_context_bsz"], collate_frame_val)
    query_loader = _loader(text_dataset, config, config["eval_query_bsz"], collate_text_val)
    return context_loader, query_loader


def build_training_loaders(config):
    train_caption, train_text_features, feature_directory, mapping_path = _collection_paths(config, "train")
    visual_features = FeatureStore(feature_directory)
    config["visual_feat_dim"] = visual_features.ndims
    video_to_frames = read_mapping(mapping_path)
    train_dataset = Dataset4PRVR(
        train_caption,
        visual_features,
        train_text_features,
        config,
        video2frames=video_to_frames,
    )
    train_loader = _loader(train_dataset, config, config["batchsize"], collate_train, shuffle=True)
    validation_loaders = _build_split(config, "val")
    return train_loader, *validation_loaders


def build_evaluation_loaders(config):
    return _build_split(config, "test")
