import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool

import jsonlines
import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

import gc
import torch

from bedrock_utils import predict_one_eg_mistral, predict_one_eg_claude_instant
from local_llm_batch import predict_one_local
from prompts_v2 import (
    ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    ZS_NAIVE_PROMPT_STR_FOR_QWEN,
    ZS_NAIVE_PROMPT_STR_FOR_LLAMA,
    ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
    ZS_KEYWORD_PROMPT_STR_FOR_QWEN,
    ZS_KEYWORD_PROMPT_STR_FOR_LLAMA,
    ZS_COT_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    ZS_COT_KEYWORD_PROMPT_STR_FOR_QWEN,
    ZS_COT_KEYWORD_PROMPT_STR_FOR_LLAMA
)

ZS_NAIVE_PROMPT_STR = {
    "mixtral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    "local_mistral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    "qwen": ZS_NAIVE_PROMPT_STR_FOR_QWEN,
    "llama": ZS_NAIVE_PROMPT_STR_FOR_LLAMA
}

ZS_KEYWORD_PROMPT_STR = {
    "mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
    "local_mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "qwen": ZS_KEYWORD_PROMPT_STR_FOR_QWEN,
    "llama": ZS_KEYWORD_PROMPT_STR_FOR_LLAMA
}

ZS_COT_PROMPT_STR = {
    "local_mistral": ZS_COT_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "qwen": ZS_COT_KEYWORD_PROMPT_STR_FOR_QWEN,
    "llama": ZS_COT_KEYWORD_PROMPT_STR_FOR_LLAMA
}

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def estimate_logits_threshold(dataset_file, percentile_threshold):
    if not os.path.exists(dataset_file):
        logging.warning("validation set not found for logits threshold.")
        return -1

    with jsonlines.open(dataset_file) as f:
        data = list(f)

    if "input_kw_model" not in data[0]:
        logging.warning("input_kw_model not found in the file. Use -1 as threshold.")
        return -1

    logits = []

    for item in data:
        for kw_info_model in item["input_kw_model"]:
            logits.append(kw_info_model["score"])
    return np.percentile(logits, percentile_threshold)


class NaivePrompt(object):
    def __init__(self, model_name, dataset_name, customized_prompt=None):
        self.prompt = customized_prompt or ZS_NAIVE_PROMPT_STR[model_name][dataset_name]

    def __call__(self, example):
        return self.prompt.replace("<text>", example["trunc_input"])


def remove_duplicate_top_k(candidates, top_k, threshold=70):
    ret = []

    for candidate in candidates:
        to_delete = set()
        to_skip = False

        if len(ret) >= top_k:
            break

        for added_kw in ret:
            if fuzz.ratio(added_kw["phrase"].lower(), candidate["phrase"].lower()) >= threshold:
                if len(added_kw["phrase"]) <= len(candidate["phrase"]):
                    to_delete.add(added_kw["phrase"])
                else:
                    to_skip = True

        ret = [item for item in ret if item["phrase"] not in to_delete]

        if not to_skip:
            ret.append(candidate)

    return ret


class SegExtTopK(object):
    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        customized_prompt=None,
    ):
        self.prompt = customized_prompt or ZS_KEYWORD_PROMPT_STR[model_name][dataset_name]
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank

    def __call__(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."
        return self.prompt.replace("<text>", example["trunc_input"]).replace("<keywords>", formatted_keywords)


class CoTPrompt(object):
    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        customized_prompt=None,
    ):
        self.prompt = customized_prompt or ZS_COT_PROMPT_STR[model_name][dataset_name]
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank

    def __call__(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."
        return self.prompt.replace("<text>", example["trunc_input"]).replace("<keywords>", formatted_keywords)

def get_prompt_fn(model_name, dataset, kw_strategy, kw_model_top_k, logits_threshold):
    if kw_strategy == "disable":
        return NaivePrompt(model_name, dataset)
    elif kw_strategy == "sigext_topk":
        return SegExtTopK(model_name, dataset, top_k=kw_model_top_k, logits_threshold=logits_threshold)
    elif kw_strategy == "cot":
        return CoTPrompt(model_name, dataset, top_k=kw_model_top_k, logits_threshold=logits_threshold)
    else:
        raise RuntimeError("unknown kw strategy.")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rouge_score(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    labels = [item["raw_output"] for item in inference_data]

    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_element = defaultdict(list)
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(target=label, prediction=pred)
        for metric_name, value in score.items():
            result_element[f"{metric_name}p"].append(value.precision)
            result_element[f"{metric_name}r"].append(value.recall)
            result_element[f"{metric_name}f"].append(value.fmeasure)

    result = {}
    for metric_name, values in result_element.items():
        result[metric_name] = np.mean(values)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def run_inference(model_name, kw_strategy, kw_model_top_k, dataset, dataset_dir, output_dir, inference_on_split="test", batch_size):
    dataset_dir = pathlib.Path(dataset_dir)
    logits_threshold = estimate_logits_threshold(str(dataset_dir.joinpath("validation.jsonl")), 75)
    logging.info(f"logits threshold is {logits_threshold}")

    if model_name == "mistral":
        predict_one_eg_fn = predict_one_eg_mistral
    elif model_name == "claude":
        predict_one_eg_fn = predict_one_eg_claude_instant
    elif model_name in ["local_mistral", "qwen", "llama"]:
        predict_one_eg_fn = predict_one_local
    else:
        raise ValueError(f"invalid model name {model_name}")

    prompting_fn = get_prompt_fn(
        model_name,
        dataset,
        kw_strategy,
        kw_model_top_k=kw_model_top_k,
        logits_threshold=logits_threshold,
    )
    assert not isinstance(prompting_fn, dict)
    dataset_dir = pathlib.Path(dataset_dir)

    dataset_filename = str(dataset_dir.joinpath(f"{inference_on_split}.jsonl"))
    with jsonlines.open(dataset_filename) as f:
        inference_data = list(f)

    output_path = pathlib.Path(output_dir).expanduser()
    os.makedirs(output_path, exist_ok=True)

    # ---- Resume logic: read partial file if exist
    partial_pred_file = output_path.joinpath(f"{inference_on_split}_predictions_partial.json")
    partial_dataset_file = output_path.joinpath(f"{inference_on_split}_dataset_partial.jsonl")

    all_res = []
    resume_idx = 0
    if partial_pred_file.exists() and partial_dataset_file.exists():
        with open(partial_pred_file) as f:
            all_res = json.load(f)
        resume_idx = len(all_res)
        logging.info(f"Resume from batch with paper {resume_idx}")

    # Loop batch
    start_from = (resume_idx // batch_size) * batch_size

    for start_idx in range(start_from, len(inference_data), batch_size):
        batch = inference_data[start_idx:start_idx + batch_size]

        # Generate prompt
        if model_name in ["local_mistral", "qwen", "llama"]:
            all_prompt = [prompting_fn(ex) for ex in batch]
        else:
            with Pool(4) as p:
                all_prompt = list(tqdm.tqdm(p.imap(prompting_fn, batch), total=len(batch)))

        for i, ex in enumerate(batch):
            if isinstance(all_prompt[i], str):
                ex["prompt_input"] = all_prompt[i]
            else:
                ex["prompt_input"] = all_prompt[i][0]
                ex["other_info"] = all_prompt[i][1]

        # Generate predictions
        if model_name in ["local_mistral", "qwen", "llama"]:
            batch_res = []
            with torch.no_grad():
                for ex in tqdm.tqdm(batch):
                    batch_res.append(predict_one_local(ex, model_name))
        else:
            with Pool(4) as p:
                batch_res = list(tqdm.tqdm(p.imap(predict_one_eg_fn, batch), total=len(batch)))

        all_res.extend(batch_res)

        # Partial save
        with jsonlines.open(partial_dataset_file, "w") as f:
            f.write_all(inference_data[:start_idx + len(batch)])

        with open(partial_pred_file, "w") as f:
            json.dump(all_res, f, indent=2)

        logging.info(f"Batch {start_idx} -> {start_idx + len(batch)} salvato.")

        cleanup_cuda()  # <<< clean GPU after every batch

    # Final save
    final_dataset_file = output_path.joinpath(f"{inference_on_split}_dataset.jsonl")
    with jsonlines.open(final_dataset_file, "w") as f:
        f.write_all(inference_data)

    final_pred_file = output_path.joinpath(f"{inference_on_split}_predictions.json")
    with open(final_pred_file, "w") as f:
        json.dump(all_res, f, indent=2)

    # Metrics
    if kw_strategy == "cot":
        summary = []
        for res in all_res:
            summary.append(res.split("Final summary:", 1)[1].strip())
        test_metrics = compute_rouge_score(inference_data, summary)
    else:
        test_metrics = compute_rouge_score(inference_data, all_res)
    with open(str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")), "w") as f:
        json.dump(test_metrics, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default="local_mistral",
                        choices=["claude", "mistral", "local_mistral", "qwen", "llama"],
                        help="llm name")
    parser.add_argument("--kw_strategy", choices=["disable", "sigext_topk", "cot"], help="keyword strategy.")
    parser.add_argument("--kw_model_top_k", default=20, type=int, help="keyword strategy.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxiv", "pubmed", "cnn", "samsum", "meetingbank"],
        help="Select from supported datasets.",
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save experiment.")
    parser.add_argument("--inference_on_split", default="test", type=str, help="split_to_run_inference")
    parser.add_argument("--batch_size", default=1, type=int, help="size of batch")

    args = parser.parse_args()

    run_inference(**vars(args))


if __name__ == "__main__":
    main()
