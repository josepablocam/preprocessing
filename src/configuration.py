CONFIGS = {}
basic = {
    # based on prior research
    "code_vocab_top_k": int(5e3),
    "nl_vocab_top_k": int(5e3),
    # mean length in github python train.function
    "code_length": 58,
    # mean length of github python train.docstrings + SO titles
    "nl_length": 30,
    "embedding_size": 300,
    # training parameters
    # based on deep code search
    "batch_size": 50,
    "margin": 0.25,
    "epochs": 100,
    "lr": 5e-4,
    "ada_weight": 0.01,
    "ada_lr": 1e-3,
    # changed to classifer since wasserstein loss seems too low on our task
    "ada_model_type": "classifier",
    "ada_hidden_size": 300,
    "fixed_embeddings": False,
    "model": "dan"
}
version2 = dict(basic)
version2["ada_model_type"] = "wasserstein"

downsampled = dict(basic)
downsampled["code_vocab_top_k"] = 67652
downsampled["nl_vocab_top_k"] = 67652

downsampled_wass = dict(downsampled)
downsampled_wass["ada_model_type"] = "wasserstein"

fixed_embeddings = dict(downsampled)
fixed_embeddings["fixed_embeddings"] = True

dan = dict(downsampled)
dan["num_dan_layers"] = 3

lstm_wass = dict(downsampled_wass)
lstm_wass["model"] = "lstm"
lstm_wass["hidden_size"] = 100
lstm_wass["bidirectional"] = True

lstm_fix = dict(lstm_wass)
lstm_fix["fixed_embeddings"] = True


CONFIGS["basic"] = basic
CONFIGS["version2"] = version2
CONFIGS["downsampled"] = downsampled
CONFIGS["downsampled-wass"] = downsampled_wass
CONFIGS["fixed-embeddings"] = fixed_embeddings
CONFIGS["dan"] = dan
CONFIGS["lstm-wass"] = lstm_wass
CONFIGS["lstm-fix"] = lstm_fix


def get_configuration(config_name):
    config = CONFIGS.get(config_name, None)
    if config is None:
        raise ValueError("Unknown configuration name")
    return config
