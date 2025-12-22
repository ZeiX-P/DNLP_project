import argparse
import logging
import pathlib
import difflib

import jsonlines
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

# === MODEL CONFIGURATIONS ===
MODELS = {
    "longformer": {
        "name": "allenai/longformer-base-4096",
        "max_length": 4096,
        "needs_block_padding": False,
        "use_explicit_mask": False,
    },
    "longformer_large": {
        "name": "allenai/longformer-large-4096",
        "max_length": 4096,
        "needs_block_padding": False,
        "use_explicit_mask": False,
    },
    "bigbird_base": {
        "name": "google/bigbird-roberta-base",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 3, 
    },
    "bigbird_large": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 3, 
    },
    # ADDED MODERNBERT CONFIGURATION
    "modernbert_large": {
        "name": "answerdotai/ModernBERT-large",
        "max_length": 8192,
        "needs_block_padding": False,
        "use_explicit_mask": False,
    },
}

def get_fuzzy_lcs_score(s1, s2):
    """
    fuzz(a, b) = |LCS(a,b)| / max(|a|, |b|)
    """
    if not s1 or not s2: return 0.0
    seq = difflib.SequenceMatcher(None, s1, s2)
    match = seq.find_longest_match(0, len(s1), 0, len(s2))
    return match.size / max(len(s1), len(s2))

class KWDatasetContext(Dataset):
    def __init__(self, dataset_filename, model_config, example_kw_hit_threshold=1, hide_gt=False):
        super().__init__()
        self.data = []
        self.model_config = model_config
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
        self.max_length = model_config["max_length"]
        

        self.needs_block_padding = model_config.get("needs_block_padding", False)
        self.block_size = model_config.get("block_size", 64)
        
        self.fuzzy_threshold = 0.7

        with jsonlines.open(dataset_filename) as f:
            self.raw_dataset = list(f)

        pos_cc = 0
        neg_cc = 0
        truncation_count = 0
        padded_short_count = 0
        
        for idx, item in tqdm.tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset), desc="process data"):

            x = [self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id]
            y = [-100]
            mask = [1]
            
            training_keywords = set()
            if not hide_gt:
                for kw in item["trunc_input_phrases"]:
                    best_score = 0.0
                    for ent in item["trunc_output_phrases"]:
                        score = get_fuzzy_lcs_score(kw["phrase"], ent["phrase"])
                        best_score = max(best_score, score)
                    if best_score >= self.fuzzy_threshold:
                        training_keywords.add(kw["phrase"])

            oracle_phrases = set()
            if not hide_gt:
                for ent in item["trunc_output_phrases"]:
                    best_match_phrase = None
                    best_match_score = 0.0
                    
                    for kw in item["trunc_input_phrases"]:
                        score = get_fuzzy_lcs_score(kw["phrase"], ent["phrase"])
                        if score > best_match_score:
                            best_match_score = score
                            best_match_phrase = kw["phrase"]
                    
                    if best_match_phrase and best_match_score >= self.fuzzy_threshold:
                        oracle_phrases.add(best_match_phrase)


            current_text_index = 0
            for kw_info in item["trunc_input_phrases"]:
                if len(x) >= self.max_length - 50: break
                
                # Filler text
                if current_text_index < kw_info["index"]:
                    content = item["trunc_input"][current_text_index: kw_info["index"]]
                    tokens = self.tokenizer(content, add_special_tokens=False)["input_ids"]
                    x.extend(tokens)
                    y.extend([-100] * len(tokens))
                    mask.extend([1] * len(tokens))
                elif current_text_index != 0: 
                    tokens = self.tokenizer(" ", add_special_tokens=False)["input_ids"]
                    x.extend(tokens)
                    y.extend([-100] * len(tokens))
                    mask.extend([1] * len(tokens))

                # Phrase tokens
                format_kw = f"{kw_info['phrase']}"
                tokens = self.tokenizer(format_kw, add_special_tokens=False)["input_ids"]
                
                label = 1 if (not hide_gt and kw_info["phrase"] in training_keywords) else 0
                
                x.extend(tokens)
                y.extend([label] * len(tokens))
                mask.extend([1] * len(tokens))
                current_text_index = kw_info["index"] + len(kw_info["phrase"])

            # Trailing text
            if current_text_index < len(item["trunc_input"]) and len(x) < self.max_length - 10:
                content = item["trunc_input"][current_text_index:]
                tokens = self.tokenizer(content, add_special_tokens=False)["input_ids"]
                x.extend(tokens)
                y.extend([-100] * len(tokens))
                mask.extend([1] * len(tokens))

            # Padding
            if self.needs_block_padding and len(x) < 1024:
                pad_len = 1024 - len(x)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                x.extend([pad_id] * pad_len)
                y.extend([-100] * pad_len)
                mask.extend([0] * pad_len)
                padded_short_count += 1

            x = x[: self.max_length - 1]
            y = y[: self.max_length - 1]
            mask = mask[: self.max_length - 1]
            
            # EOS/SEP Token
            if self.tokenizer.sep_token_id is not None:
                x.extend([self.tokenizer.sep_token_id])
            elif self.tokenizer.eos_token_id is not None:
                x.extend([self.tokenizer.eos_token_id])
            else:
                x.extend([2]) # Fallback
                
            y.extend([-100])
            mask.extend([1])

            if self.needs_block_padding:
                rem = len(x) % self.block_size
                if rem != 0:
                    pad_len = self.block_size - rem
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    x.extend([pad_id] * pad_len)
                    y.extend([-100] * pad_len)
                    mask.extend([0] * pad_len)

            x = torch.tensor(x).long()
            y = torch.tensor(y).long()
            mask = torch.tensor(mask).long()

            # Pack Oracle Phrases as a string for validation decoding
            oracle_str = "|||".join(list(oracle_phrases))
            
            self.data.append((x, y, mask, oracle_str))
            pos_cc += torch.sum(y == 1).numpy()
            neg_cc += torch.sum(y == 0).numpy()

        if truncation_count > 0:
            logging.warning(f"Truncated {truncation_count} examples")
        
        logging.info(f"Keyword ratio: {pos_cc / (pos_cc + neg_cc):.4f}")
        logging.info(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KeywordExtractorClf(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        model_name = model_config["name"]
        logging.info(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = 2
        
        if "num_random_blocks" in model_config and hasattr(config, "num_random_blocks"):
            config.num_random_blocks = model_config["num_random_blocks"]

        self.clf = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        
        self.clf.gradient_checkpointing_enable()
        if hasattr(self.clf.config, "use_cache"):
            self.clf.config.use_cache = False
            
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        x, y, idx = batch
        assert x.size(0) == 1
        logits = F.log_softmax(self.clf(x)[0], dim=-1)

        kw_count = 0
        score_and_label = []

        current_logits = 0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                current_logits += logits[0][i][1]
                current_len += 1
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    score_and_label.append({"kw_index": kw_count, "score": float(current_logits / current_len)})
                    kw_count += 1
                    current_logits = 0
                    current_len = 0

        return {"id": int(idx[0]), "score": score_and_label}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        
        if self.model_config.get("use_explicit_mask", False):
            logits = self.clf(input_ids=x, attention_mask=mask)[0]
        else:
            outputs = self.clf(input_ids=x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        loss_weight = torch.tensor([0.1, 1.0], device=self.device).type_as(logits)
        loss = F.cross_entropy(logits[0], y[0], reduction="sum", weight=loss_weight)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, oracle_strs = batch
        
        if self.model_config.get("use_explicit_mask", False):
            logits = self.clf(input_ids=x, attention_mask=mask)[0]
        else:
            outputs = self.clf(input_ids=x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        loss_weight = torch.tensor([0.1, 1.0], device=self.device).type_as(logits)
        loss = F.cross_entropy(logits[0], y[0], reduction="sum", weight=loss_weight).cpu().float().numpy()
        
        logits = F.log_softmax(logits, dim=-1)
        logits_np = logits.cpu().float().numpy()
        y_np = y.cpu().numpy()
        x_np = x.cpu().numpy()
        
      
        candidates = []
        current_logits = 0
        current_len = 0
        current_in_phrase = False
        current_token_ids = []

        for i in range(x.size(1)):
            if y_np[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                    current_token_ids = []
                current_logits += logits_np[0][i][1]
                current_len += 1
                current_token_ids.append(x_np[0][i])
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    candidates.append({
                        "score": float(current_logits / current_len),
                        "tokens": tuple(current_token_ids),
                    })
                    current_logits = 0
                    current_len = 0
                    current_token_ids = []

     
        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
        
        if not hasattr(self, 'val_tokenizer'):
             self.val_tokenizer = AutoTokenizer.from_pretrained(self.model_config["name"], trust_remote_code=True)
        
        final_preds = []
        
        for cand in candidates:
            cand_text = self.val_tokenizer.decode(list(cand["tokens"]), skip_special_tokens=True).strip()
            if not cand_text: continue

            is_duplicate = False
            for i, existing in enumerate(final_preds):
                if get_fuzzy_lcs_score(cand_text, existing["text"]) >= 0.7:
                    is_duplicate = True
                    if len(cand_text) > len(existing["text"]):
                        final_preds[i] = {"text": cand_text, "score": cand["score"]}
                    break
            
            if not is_duplicate:
                final_preds.append({"text": cand_text, "score": cand["score"]})
                
            if len(final_preds) >= 50: 
                break

       
        oracle_set = set(oracle_strs[0].split("|||")) if len(oracle_strs[0]) > 0 else set()
        total_oracle_positives = len(oracle_set)
        
        step_output = {"loss": loss}
        
        for top_k in [10, 15, 20, 35, 40]:
            if len(final_preds) > 0:
                current_top_k = final_preds[:top_k]
                matched_oracles = set()
                
                for pred in current_top_k:
                    for oracle_phrase in oracle_set:
                        if get_fuzzy_lcs_score(oracle_phrase, pred["text"]) >= 0.7:
                            matched_oracles.add(oracle_phrase)
                
                hits = len(matched_oracles)
                
                step_output[f"recall_{top_k}"] = hits / max(total_oracle_positives, 1)
                step_output[f"precision_{top_k}"] = hits / max(len(current_top_k), 1)
            else:
                step_output[f"recall_{top_k}"] = 0
                step_output[f"precision_{top_k}"] = 0
                
        self.validation_step_outputs.append(step_output)

    def on_validation_epoch_end(self):
        for top_k in [10, 15, 20, 35, 40]:
            self.log(f"val/precision_{top_k}", float(np.mean([item[f"precision_{top_k}"] for item in self.validation_step_outputs])), sync_dist=True)
            self.log(f"val/recall_{top_k}", float(np.mean([item[f"recall_{top_k}"] for item in self.validation_step_outputs])), sync_dist=True)
        self.log("val/loss", float(np.mean([item["loss"] for item in self.validation_step_outputs])), sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.estimated_stepping_batches, num_warmup_steps=100
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["longformer", "longformer_large","bigbird_base","bigbird_large", "modernbert_large"])
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--precision", default="bf16-true")
 
    parser.add_argument("--recall_k", type=int, default=20, help="Recall@K to monitor (default 20)")

    args = parser.parse_args()
    model_config = MODELS[args.model]
    
    pl.seed_everything(42)
    train_set = KWDatasetContext(str(pathlib.Path(args.dataset_dir).joinpath("train.jsonl")), model_config)
    val_set = KWDatasetContext(str(pathlib.Path(args.dataset_dir).joinpath("validation.jsonl")), model_config)
    

    monitor_metric = f"val/recall_{args.recall_k}"
    logging.info(f"Checkpointing will monitor: {monitor_metric}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        monitor=monitor_metric,
        mode="max",
        every_n_epochs=1,
        filename=f"epoch_{{epoch:02d}}-step_{{step:06d}}-recall{args.recall_k}_{{{monitor_metric}:.3f}}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        gradient_clip_val=1,
        accumulate_grad_batches=32,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    
    trainer.fit(KeywordExtractorClf(model_config), DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0))


if __name__ == "__main__":
    main()