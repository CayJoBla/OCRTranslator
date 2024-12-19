# train_steel.py

from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    default_data_collator,
)
from datasets import load_dataset
import evaluate
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFont
import argparse
import numpy as np
import random

from translate import english_to_steel_phonetics
from render_text import text_to_image


# Custom dataset class for data preprocessing
class SteelOCRDataset(Dataset):
    FONT_PATHS = [
        "/home/cayjobla/OCRTranslator/AlloyofLaw/AlloyofLawExpanded.ttf",
        "/home/cayjobla/OCRTranslator/AlloyofLaw/AlloyofLaw-BoldExpanded.ttf",
    ]

    def __init__(
        self, 
        data, 
        processor,
        font_size=50, 
        min_char_length=10,
        max_words_per_chunk=8,
        max_target_length=128,
    ):
        self._data = data
        self.translator = english_to_steel_phonetics
        self.processor = processor
        self.font_size = font_size
        self.min_char_length = min_char_length
        self.max_target_length = max_target_length
        self.max_words_per_chunk = max_words_per_chunk
        self.fonts = [
            ImageFont.truetype(path, size=font_size) for path in self.FONT_PATHS
        ]
        self._data = self._data.filter(
            lambda x: len(x["text"]) > self.min_char_length
        )
        self._data = self._data.map(
            self._preprocess_data, batched=True, batch_size=1000
        )

    def __len__(self):
        return len(self._data)

    @property
    def font(self):
        return np.random.choice(self.fonts)

    def _preprocess_data(self, batch):
        # Sample a chunk of text from each example
        chunk_size = 3 * self.max_target_length
        chunks = []
        translated_chunks = []
        images = []
        for example in batch['text']:   # Split each example, translate, and render text
            idx = 0
            if len(example) > chunk_size:
                idx = random.randint(0, len(example) - chunk_size)
            words = example[idx:idx+chunk_size].split()[1:-1] # Drop start and end partial words
            words = [word for word in words if len(word) > 0]
            if len(words) > 0:
                chunks.append(" ".join(words[:self.max_words_per_chunk]))
                translated_chunks.append(self.translator(chunks[-1]))
                images.append(text_to_image(translated_chunks[-1], font=self.font))
        
        return {'text': chunks, 'steel': translated_chunks, 'image': images}

    def __getitem__(self, idx):
        # Get the image and text
        image = self._data[idx]["image"]
        if "text" in self._data[idx]:
            text = self._data[idx]["text"]
        elif "english" in self._data[idx]:
            text = self._data[idx]["english"]
        else:
            text = None
        
        # Preprocessing
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = None
        if text is not None:
            labels = self.processor.tokenizer(
                text, 
                padding="max_length",
                max_length=self.max_target_length,
                return_tensors="pt"
            ).input_ids

            # Ignore the padding tokens
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels = labels.squeeze()

        return {"pixel_values": pixel_values.squeeze(), "labels": labels}


def train(
    pretrained_model_name_or_path="microsoft/trocr-base-stage1", 
    dataset_path=None, 
    dataset_args=None,  # I don't like this parsing method
    max_length=128,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4,
    batch_size=8,
    num_epochs=1,
    logging_steps=100,
    save_steps=500,
    eval_steps=1000,
    output_dir=None,
    run_name=None,
):
    # Load the pretrained model and processor
    processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

    # Load the text dataset (should have a "text" column)
    if dataset_path is None:
        raise ValueError("Must provide a dataset path.")
    dataset_args = dataset_args or []
    dataset = load_dataset(dataset_path, *dataset_args)
    if "test" not in dataset:   
        dataset = dataset.train_test_split(test_size=0.2)

    # # Initialize the train and test datasets
    # train_dataset = SteelOCRDataset(
    #     dataset["train"], processor, max_target_length=max_length
    # )
    # val_dataset = SteelOCRDataset(
    #     dataset["test"], processor, max_target_length=max_length
    # )

    # Downsample the data for testing
    train_dataset = SteelOCRDataset(
        dataset["train"].train_test_split(train_size=.1)["train"],
        processor,
        max_target_length=max_length
    )
    val_dataset = SteelOCRDataset(
        dataset["test"].train_test_split(train_size=.1)["train"],
        processor,
        max_target_length=max_length
    )

    print("Train dataset length:", len(train_dataset))
    print("Val dataset length:", len(val_dataset))

    # Set model config params
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.config.max_length = max_length
    model.config.early_stopping = early_stopping
    model.config.no_repeat_ngram_size = no_repeat_ngram_size
    model.config.length_penalty = length_penalty
    model.config.num_beams = num_beams

    # Load the evalutation metric (CER)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        fp16=True, 
        output_dir=output_dir or "trocr-temp",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to="wandb",
        run_name=run_name
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator
    )

    trainer.train()

    if output_dir is not None:
        trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VisionEncoderDecoder model.")

    parser.add_argument(
        "--pretrained_model_name_or_path", 
        "--model_name_or_path",
        "--model",
        type=str, 
        default="microsoft/trocr-base-stage1", 
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--dataset_path", 
        "--dataset",
        type=str, 
        required=True, 
        help="Path to the dataset."
    )
    parser.add_argument(
        "--dataset_args",
        nargs="*",
        help="Additional arguments to pass to the dataset loader."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=64, 
        help="Maximum length of the target text and model generation."
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False, 
        help="Whether to use early stopping."
    )
    parser.add_argument(
        "--no_repeat_ngram_size", 
        type=int, 
        default=3, 
        help="No repeat ngram size."
    )
    parser.add_argument(
        "--length_penalty", 
        type=float, 
        default=2.0, 
        help="Length penalty."
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=4, 
        help="Number of beams for beam search."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100, 
        help="Log every X training steps."
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500, 
        help="Save checkpoint every X training steps."
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=1000, 
        help="Run an evaluation every X training steps."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="The name to give the wandb run."
    )
    
    args = parser.parse_args()
    train(**vars(args))