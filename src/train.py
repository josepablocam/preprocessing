import argparse
from collections import defaultdict
import copy
import datetime
import json
import os
import pickle

import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.utils.data
import tqdm

from . import data
from . import code_search_model
from . import precomputed_embeddings
from . import utils

outputManager = 0

BATCH_SIZE = 50
MARGIN = 0.25
NUM_EPOCHS = 100
LR = 5e-4
HIDDEN_SIZE = 300
NUM_LAYERS = 3 # For DAN
DROPOUT = 0.0 # For DAN
DAN_HIDDEN_SIZE = 2*HIDDEN_SIZE


def load_data(batch_size, code_path, docstring_path):
    code = np.load(code_path).astype(np.int64)
    docstrings = np.load(docstring_path).astype(np.int64)

    if utils.have_gpu():
        code = torch.tensor(code).cuda()
        docstrings = torch.tensor(docstrings).cuda()

    paired_dataset = data.CodeSearchDataset(code, docstrings)
    paired_dl = torch.utils.data.DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return paired_dl


def log_losses(named_losses, batch_count, print_every, tensorboard_writer):
    summary = {}
    for loss_name, loss_values in named_losses.items():
        mean_loss = np.mean(loss_values)
        summary[loss_name] = mean_loss
        # log training info for tensorboard
        tensorboard_writer.add_scalar(
            "train/{}".format(loss_name), mean_loss, batch_count
        )

    if batch_count % print_every == 0:
        outputManager.say("Batch: {}, {}".format(batch_count, summary))
        # reset
        named_losses = {name: [] for name in named_losses.keys()}
    return named_losses


def log_models(named_models, model_folder, epoch, time_seconds=None):
    for model_name, model in named_models.items():
        save_model(
            model,
            model_folder,
            model_name,
            epoch,
            time_seconds,
            symlink_latest=True
        )
    return named_models


def save_model(
        model,
        model_folder,
        model_name,
        epoch,
        time_seconds=None,
        symlink_latest=True,
):
    model_checkpoint_path = os.path.join(
        model_folder,
        "{}_epoch_{}.pth".format(model_name, epoch),
    )
    torch.save(model, model_checkpoint_path)

    # write training time associated with this model
    if time_seconds is not None:
        model_time_file = os.path.join(model_folder, "timestamps.csv")
        file_mode = "a" if os.path.exists(model_time_file) else "w"
        with open(model_time_file, file_mode) as fout:
            fout.write("{},{}\n".format(model_checkpoint_path, time_seconds))

    if symlink_latest:
        symlink_name = "{}_latest".format(model_name)
        symlink_path = os.path.join(model_folder, symlink_name)
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        # need to point to absolute location
        os.symlink(os.path.abspath(model_checkpoint_path), symlink_path)


def train(
        code_path,
        docstrings_path,
        embeddings_path,
        vocab_encoder_path,
        model_option,
        print_every=10,
        save_every=1,
        output_folder=None,
        batch_size=BATCH_SIZE,
        lr=LR,
        margin=MARGIN,
        num_epochs=NUM_EPOCHS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, # for dan
        dropout=DROPOUT, # for dan
        dan_hidden_size=DAN_HIDDEN_SIZE,
        fixed_embeddings=True,
        valid_code_path=None,
        valid_docstrings_path=None,
):
    embeddings = precomputed_embeddings.read_embeddings(embeddings_path)
    emb_size = list(embeddings.values())[0].shape[0]

    with open(vocab_encoder_path, "rb") as fin:
        vocab_encoder = pickle.load(fin)

    vocab_size = len(vocab_encoder.vocab_map)

    # folder name to save down stuff...
    if output_folder:
        run_folder = output_folder
    else:
        run_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # folder to save down model checkpoints
    model_folder = os.path.join(run_folder, "models")
    utils.create_dir(model_folder)
    # save down the configuration for reference
    config = {
        "code_path": code_path,
        "docstrings_path": docstrings_path,
        "embeddings_path": embeddings_path,
        "vocab_encoder_path": vocab_encoder_path,
        "output_folder": output_folder,
        "batch_size": batch_size,
        "lr": lr,
        "margin": margin,
        "num_epochs": num_epochs,
        "fixed_embeddings": fixed_embeddings,
        "valid_code_path": valid_code_path,
        "valid_docstrings_path": valid_docstrings_path,
        "model": model_option,
        "dan_hidden_size": dan_hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    with open(os.path.join(model_folder, "config.json"), "w") as fout:
        json.dump(config, fout)

    log_dir = os.path.join(run_folder, "runs")
    utils.create_dir(log_dir)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    global outputManager
    outputManager = utils.OutputManager(log_dir)

    paired_dl = load_data(
        batch_size,
        code_path,
        docstrings_path,
    )
    use_validation = (
        valid_code_path is not None and valid_docstrings_path is not None
    )
    if use_validation:
        valid_dl = load_data(
            batch_size,
            valid_code_path,
            valid_docstrings_path,
        )
    if model_option == 'lstm':
        sim_model = code_search_model.LSTMModel(
            margin,
            vocab_size,  # same vocab for both code/NL
            vocab_size,
            emb_size,
            hidden_size=hidden_size,
            bidirectional=True,
            fixed_embeddings=fixed_embeddings,
            same_embedding_fun=False,
        )
    else:
        sim_model = code_search_model.DANModel(
            margin,
            vocab_size, # same vocab for both code/NL
            vocab_size,
            emb_size,
            hidden_size=dan_hidden_size,
            dropout=dropout,
            fixed_embeddings=fixed_embeddings,
            num_layers=num_layers,
        )
    code_search_model.setup_precomputed_embeddings(
        sim_model,
        vocab_encoder_path,
        embeddings_path,
    )

    if utils.have_gpu():
        sim_model = sim_model.cuda()

    sim_optimizer = optim.Adam(utils.get_trainable_params(sim_model), lr)

    models = {}
    models["sim_model"] = sim_model

    log_models(models, model_folder, "pre-start")
    batch_count = 0
    losses = defaultdict(lambda: [])

    if use_validation:
        best_valid_loss = None
        best_valid_model = None

    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        outputManager.say("Epoch:{}".format(epoch))
        for code, docstrings, fake_code in tqdm.tqdm(paired_dl):
            sim_optimizer.zero_grad()

            sim_losses = sim_model.losses(code, docstrings, fake_code)
            total_loss = sim_losses.mean()
            losses["total_loss"].append(total_loss.item())
            total_loss.backward()
            sim_optimizer.step()

            losses = log_losses(
                losses,
                batch_count,
                print_every,
                tensorboard_writer,
            )
            batch_count += 1

        if use_validation:
            loss = evaluate_loss(sim_model, valid_dl)
            if best_valid_loss is None or loss < best_valid_loss:
                print(
                    "Found better model ({} < {}) @ epoch {}".format(
                        loss,
                        best_valid_loss,
                        epoch,
                    )
                )
                best_valid_loss = loss
                best_valid_model = copy.deepcopy(sim_model)
                # make sure to log it each time, just in case
                log_models(
                    {
                        "sim_model": best_valid_model
                    },
                    model_folder,
                    "best",
                    time_seconds=None,
                )

        if epoch % save_every == 0:
            current_time = datetime.datetime.now()
            amt_time_seconds = (current_time - start_time).total_seconds()
            log_models(models, model_folder, epoch, amt_time_seconds)

    # final log if any
    log_models(models, model_folder, "final", amt_time_seconds)

    if use_validation:
        log_models(
            {
                "sim_model": best_valid_model
            },
            model_folder,
            "best",
            amt_time_seconds,
        )


def evaluate_loss(model, dl):
    model.eval()

    valid_losses = []
    for code, docstring, fake_code in dl:
        losses = model.losses(code, docstring, fake_code)
        valid_losses.extend(losses.tolist())

    # turn back on training
    model.train()
    return np.mean(valid_losses)


def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "-c", "--code_path", type=str, help="Path to github code .npy file"
    )
    parser.add_argument(
        "-d",
        "--docstrings_path",
        type=str,
        help="Path to github docstrings .npy file"
    )
    parser.add_argument(
        "-e",
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to precomputed embeddings .vec file",
    )
    parser.add_argument(
        "-v",
        "--vocab_encoder_path",
        type=str,
        default=None,
        help="Path to vocabulary encoder .pkl file",
    )
    parser.add_argument(
        "-p",
        "--print_every",
        type=int,
        default=1000,
        help="Print losses every N batches"
    )
    parser.add_argument(
        "-s",
        "--save_every",
        type=int,
        default=1,
        help="Save model every N epochs",
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        type=str,
        help="Directory name to save execution info in models/ and runs/",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train(
        args.code_path,
        args.docstrings_path,
        embeddings_path=args.embeddings_path,
        vocab_encoder_path=args.vocab_encoder_path,
        print_every=args.print_every,
        save_every=args.save_every,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
