from transformers.models.auto.tokenization_auto import AutoTokenizer


def convert_to_features(
#    example_batch, model_name="prajjwal1/bert-mini", max_length=512
    example_batch, model_name="EMBO/BioMegatron345mCased", max_length=512
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = list(example_batch["sentence"])

    features = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = example_batch["target"]
    return features
