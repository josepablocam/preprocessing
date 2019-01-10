import argparse
from collections import defaultdict
import datetime
import json
import os

import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.utils.data
import tqdm

from . import configuration
from . import data
from . import code_search_model
from . import domain_adaptation_model
from . import utils

outputManager = 0


def load_data(batch_size, code_path, docstring_path, query_path, answers_path):
    code = np.load(code_path).astype(np.int64)
    docstrings = np.load(docstring_path).astype(np.int64)

    if utils.have_gpu():
        code = torch.tensor(code).cuda()
        docstrings = torch.tensor(docstrings).cuda()

    paired_dataset = data.CodeSearchDataset(code, docstrings)

    # data loaders
    paired_dl = torch.utils.data.DataLoader(
        paired_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
    )
    if query_path is not None:
        queries = np.load(query_path).astype(np.int64)
        if utils.have_gpu():
            queries = torch.tensor(queries).cuda()
        adversarial_nl_dataset = data.CrossDomainDataset(
            docstrings,
            queries,
        )
        adversarial_nl_dl = torch.utils.data.DataLoader(
            adversarial_nl_dataset,
            batch_size=batch_size * 2,
            shuffle=True,
            drop_last=True,
        )
    else:
        adversarial_nl_dl = None

    if answers_path is not None:
        answers = np.load(answers_path).astype(np.int64)
        if utils.have_gpu():
            answers = torch.tensor(answers).cuda()
        adversarial_code_dataset = data.CrossDomainDataset(
            code,
            answers,
        )
        adversarial_code_dl = torch.utils.data.DataLoader(
            adversarial_code_dataset,
            batch_size=batch_size * 2,
            shuffle=True,
            drop_last=True,
        )
    else:
        adversarial_code_dl = None

    return paired_dl, adversarial_nl_dl, adversarial_code_dl


def log_losses(named_losses, batch_count, print_every, tensorboard_writer):
    summary = {}
    for loss_name, loss_values in named_losses.items():
        mean_loss = np.mean(loss_values)
        summary[loss_name] = mean_loss
        # log training info for tensorboard
        tensorboard_writer.add_scalar("train/{}".format(loss_name), mean_loss,
                                      batch_count)

    if batch_count % print_every == 0:
        outputManager.say("Batch: {}, {}".format(batch_count, summary))
        # reset
        named_losses = {name: [] for name in named_losses.keys()}
    return named_losses


def log_models(named_models, folder, epoch, time_seconds=None):
    for model_name, model in named_models.items():
        save_model(
            model,
            folder,
            model_name,
            epoch,
            time_seconds,
            symlink_latest=True)
    return named_models


def save_model(
        model,
        run_folder,
        model_name,
        epoch,
        time_seconds=None,
        symlink_latest=True,
):
    model_checkpoint_path = os.path.join(
        "models",
        run_folder,
        "{}_epoch_{}.pth".format(model_name, epoch),
    )
    torch.save(model, model_checkpoint_path)

    # write training time associated with this model
    if time_seconds is not None:
        model_time_file = os.path.join("models", run_folder, "timestamps.csv")
        file_mode = "a" if os.path.exists(model_time_file) else "w"
        with open(model_time_file, file_mode) as fout:
            fout.write("{},{}\n".format(model_checkpoint_path, time_seconds))

    if symlink_latest:
        symlink_name = "{}_latest".format(model_name)
        symlink_path = os.path.join("models", run_folder, symlink_name)
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        # need to point to absolute location
        os.symlink(os.path.abspath(model_checkpoint_path), symlink_path)


def train(
        config,
        code_path,
        docstrings_path,
        queries_path=None,
        answers_path=None,
        embeddings_path=True,
        code_vocab_encoder_path=None,
        nl_vocab_encoder_path=None,
        same_embedding_fun=False,
        print_every=1000,
        save_every=1,
        output_folder=None,
):
    # model parameters
    margin = config["margin"]
    code_vocab_size = config["code_vocab_top_k"]
    nl_vocab_size = config["nl_vocab_top_k"]
    emb_size = config["embedding_size"]
    fixed_embeddings = config.get("fixed_embeddings", False)
    model_type = config["model"]
    if model_type == "dan":
        num_dan_layers = config.get("num_dan_layers", 0)

    # training parameters
    batch_size = config["batch_size"]
    lr = config["lr"]
    num_epochs = config["epochs"]

    # folder name to save down stuff...
    if output_folder:
        run_folder = output_folder
    else:
        run_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # folder to save down model checkpoints
    model_folder = os.path.join("models", run_folder)
    utils.create_dir(model_folder)
    # save down the configuration for reference
    with open(os.path.join(model_folder, "config.json"), "w") as fout:
        json.dump(config, fout)

    tensorboard_writer = SummaryWriter(
        log_dir=os.path.join("runs", run_folder))
    global outputManager
    outputManager = utils.OutputManager(os.path.join("runs", run_folder))

    paired_dl, nl_ada_dl, code_ada_dl = load_data(
        batch_size,
        code_path,
        docstrings_path,
        queries_path,
        answers_path,
    )
    if model_type == "dan":
        sim_model = code_search_model.CodeDocstringModel(
            margin,
            code_vocab_size,
            nl_vocab_size,
            emb_size,
            fixed_embeddings=fixed_embeddings,
            num_dan_layers=num_dan_layers,
            same_embedding_fun=same_embedding_fun,
        )
    elif model_type == "lstm":
        sim_model = code_search_model.LSTMModel(
            margin,
            code_vocab_size,
            nl_vocab_size,
            emb_size,
            config["hidden_size"],
            bidirectional=config["bidirectional"],
            fixed_embeddings=fixed_embeddings,
            same_embedding_fun=same_embedding_fun,
        )
    if embeddings_path is not None:
        code_search_model.setup_precomputed_embeddings(
            sim_model,
            code_vocab_encoder_path,
            nl_vocab_encoder_path,
            embeddings_path,
        )

    if utils.have_gpu():
        sim_model = sim_model.cuda()

    sim_optimizer = optim.Adam(utils.get_trainable_params(sim_model), lr)

    models = {}
    models["sim_model"] = sim_model

    use_adversarial_queries = queries_path is not None
    use_adversarial_code = answers_path is not None

    if use_adversarial_queries:
        # adversarial queries
        nl_ada_wrapper = domain_adaptation_model.AdversarialWrapper(
            config,
            sim_model.output_size,
            nl_ada_dl,
            sim_model.embed_nl,
        )
        models["nl_ada_model"] = nl_ada_wrapper.model

    if use_adversarial_code:
        # adversarial code
        code_ada_wrapper = domain_adaptation_model.AdversarialWrapper(
            config,
            sim_model.output_size,
            code_ada_dl,
            sim_model.embed_code,
        )
        models["code_ada_model"] = code_ada_wrapper.model

    log_models(models, run_folder, "pre-start")
    batch_count = 0
    losses = defaultdict(lambda: [])

    start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        outputManager.say("Epoch:{}".format(epoch))
        for code, docstrings, fake_docstrings in tqdm.tqdm(paired_dl):
            if use_adversarial_queries:
                nl_ada_wrapper.zero_grad()

            if use_adversarial_code:
                code_ada_wrapper.zero_grad()

            sim_optimizer.zero_grad()

            sim_losses = sim_model.losses(code, docstrings, fake_docstrings)
            total_loss = sim_losses.mean()
            losses["sim_loss"].append(total_loss.item())

            if use_adversarial_queries:
                nl_ada_loss, nl_ada_weight = nl_ada_wrapper.train_iteration()
                losses["nl_ada_loss"] = nl_ada_loss.item()
                total_loss -= nl_ada_loss * nl_ada_weight

            if use_adversarial_code:
                code_ada_loss, code_ada_weight = code_ada_wrapper.train_iteration()
                losses["code_ada_loss"] = code_ada_loss.item()
                total_loss -= code_ada_loss * code_ada_weight

            losses["total_loss"] = total_loss.item()
            # do single backward for performance reasons
            total_loss.backward()
            sim_optimizer.step()

            if use_adversarial_queries:
                nl_ada_wrapper.step()

            if use_adversarial_code:
                code_ada_wrapper.step()

            losses = log_losses(
                losses,
                batch_count,
                print_every,
                tensorboard_writer,
            )
            batch_count += 1

        if batch_count % save_every == 0:
            current_time = datetime.datetime.now()
            amt_time_seconds = (current_time - start_time).total_seconds()
            log_models(models, run_folder, epoch, amt_time_seconds)


def update_configuration(config, new_options_str):
    # copy first
    config = dict(config)
    new_options = [o.split(":") for o in new_options_str.split(",")]
    for k, new_v in new_options:
        if k not in config:
            raise ValueError("Cannot overwrite nonexisting config option")
        _type = type(config[k])
        if _type == bool:
            typed_new_v = new_v == "True"
        else:
            typed_new_v = _type(new_v)
        config[k] = typed_new_v
    return config


def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("config", type=str, help="Name of configuration")
    parser.add_argument(
        "-c", "--code_path", type=str, help="Path to github code .npy file")
    parser.add_argument(
        "-d",
        "--docstrings_path",
        type=str,
        help="Path to github docstrings .npy file")
    parser.add_argument(
        "-e",
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to precomputed embeddings .vec file",
    )
    parser.add_argument(
        "-cv",
        "--code_vocab_encoder_path",
        type=str,
        default=None,
        help="Path to code vocabulary encoder .pkl file",
    )
    parser.add_argument(
        "-nv",
        "--nl_vocab_encoder_path",
        type=str,
        default=None,
        help="Path to NL vocabulary encoder .pkl fil",
    )
    # used for adversarial training
    parser.add_argument(
        "-q",
        "--queries_path",
        type=str,
        default=None,
        help="Path to adversarial SO queries .npy file",
    )
    parser.add_argument(
        "-a",
        "--answers_path",
        type=str,
        default=None,
        help="Path to adversarial SO answers .npy file",
    )
    parser.add_argument(
        "--same_embedding_fun",
        action="store_true",
        help="Use same embedding function for code and NL",
    )
    parser.add_argument(
        "-o",
        "--overwrite_config",
        type=str,
        default=None,
        help="""
        comma-separate list of configuration options,
        format should be <key-name>:<value>
        """,
    )
    parser.add_argument(
        "-p",
        "--print_every",
        type=int,
        default=1000,
        help="Print losses every N batches")
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
    parser.add_argument(
        "--ada_lr",
        type=float,
        help="Learning rate for the adversarial part",
    )
    args = parser.parse_args()

    embeddings_args = [
        args.embeddings_path,
        args.code_vocab_encoder_path,
        args.nl_vocab_encoder_path,
    ]
    embeddings_args_defined = sum(e is not None for e in embeddings_args)
    if (embeddings_args_defined != 0 and embeddings_args_defined != 3):
        raise ValueError("""
            embeddings_path, code_vocab_encoder_path, nl_vocab_encoder_path
            are all required if one is defined
            """)

    return args


def main():
    args = get_args()
    config = configuration.get_configuration(args.config)
    if args.overwrite_config is not None:
        config = update_configuration(config, args.overwrite_config)
    # record the command line arguments in the config, so we have when saved
    config.update(vars(args))

    train(
        config,
        args.code_path,
        args.docstrings_path,
        queries_path=args.queries_path,
        answers_path=args.answers_path,
        embeddings_path=args.embeddings_path,
        code_vocab_encoder_path=args.code_vocab_encoder_path,
        nl_vocab_encoder_path=args.nl_vocab_encoder_path,
        same_embedding_fun=args.same_embedding_fun,
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
