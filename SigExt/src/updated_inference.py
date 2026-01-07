import argparse
import copy
import glob
import logging
import pathlib
import jsonlines
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from train_longformer_extractor_context import KWDatasetContext, KeywordExtractorClf, MODELS


def find_best_checkpoint(checkpoint_dir, recall_k=20):
    candidates = sorted(glob.glob(f"{checkpoint_dir}/lightning_logs/*/checkpoints/*.ckpt"))
    if not len(candidates):
        raise RuntimeError("Candidates not found.")
    best_score = 0
    best_checkpoint = None
    for item in candidates:
        tokens = pathlib.Path(item).stem.split("-")
        info = {}
        for tok in tokens:
            parts = tok.split("_", 1)  # Split only on first underscore
            if len(parts) == 2:
                key, value = parts
                info[key] = value
        recall_key = f"recall{recall_k}"
        if recall_key in info and float(info[recall_key]) >= best_score:
            best_checkpoint = item
            best_score = float(info[recall_key])
    return best_checkpoint


def parse_result(raw_dataset, predicts):
    raw_dataset = copy.deepcopy(raw_dataset)
    for pred_info in predicts:
        example_info = raw_dataset[pred_info["id"]]
        score = pred_info["score"]
        if len(score) > len(example_info["trunc_input_phrases"]):
            raise RuntimeError("model prediction does not match with number of phrases.")
        example_info["input_kw_model"] = score
    return raw_dataset


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Run model inference")
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="directory of checkpoints.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save predictions.")
    parser.add_argument("--model", required=True, type=str, choices=list(MODELS.keys()), help="Model configuration to use.")
    parser.add_argument("--recall_k", type=int, default=20, help="Recall@K used for checkpoint selection.")
    args = parser.parse_args()

    model_config = MODELS[args.model]
    logging.info(f"Using model config: {args.model}")
    logging.info(f"Config details: {model_config}")

    best_checkpoint = find_best_checkpoint(
        str(pathlib.Path(args.checkpoint_dir).expanduser()),
        recall_k=args.recall_k
    )
    logging.info(f"Using checkpoint: {best_checkpoint}")

    model = KeywordExtractorClf.load_from_checkpoint(best_checkpoint, model_config=model_config)

    trainer = pl.Trainer(devices=1, accelerator="gpu")

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["validation", "test"]:
        dataset = KWDatasetContext(
            dataset_filename=str(dataset_dir.joinpath(f"{split}.jsonl")),
            model_config=model_config,
            example_kw_hit_threshold=0,
            hide_gt=True,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        predicts = trainer.predict(model, dataloaders=dataloader)
        dataset_with_prediction = parse_result(dataset.raw_dataset, predicts)
        with jsonlines.open(str(output_dir.joinpath(f"{split}.jsonl")), "w") as f:
            f.write_all(dataset_with_prediction)


if __name__ == "__main__":
    main()