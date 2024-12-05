# train.py

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
from PIL import Image
import argparse


# Custom dataset class for data preprocessing
class VisionEncoderDecoderDataset(Dataset):
    def __init__(self, data, processor, max_target_length=128):
        self._data = data
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self._data)

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
):
    # Load the pretrained model and processor
    processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

    # Load the dataset of images and translations
    if dataset_path is None:
        raise ValueError("Must provide a dataset path.")
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.train_test_split(test_size=0.2)

    # Initialize the train and test datasets
    # train_dataset = VisionEncoderDecoderDataset(
    #     dataset["train"], processor, max_target_length=max_length
    # )
    # test_dataset = VisionEncoderDecoderDataset(
    #     dataset["test"], processor, max_target_length=max_length
    # )

    # TODO: SWAP FOR THE ABOVE
    # Downsample the data for testing
    train_dataset = VisionEncoderDecoderDataset(
        dataset["train"].train_test_split(train_size=32)["train"], 
        processor
    )
    test_dataset = VisionEncoderDecoderDataset(
        dataset["test"].train_test_split(train_size=16)["train"], 
        processor
    )

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
        output_dir=output_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
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
    
    args = parser.parse_args()
    train(**vars(args))