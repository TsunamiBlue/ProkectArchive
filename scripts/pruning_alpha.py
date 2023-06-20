import argparse, os, sys, glob, datetime
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import time

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

from omegaconf import OmegaConf

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # Windows backend
criterion_mse = torch.nn.MSELoss()
criterion_ce = torch.nn.CrossEntropyLoss()


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--datadir_in_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Prepend the final directory in the data_root to the output directory name")

    parser.add_argument("--actual_resume",
                        type=str,
                        required=True,
                        help="Path to model to actually resume from")

    parser.add_argument("--data_root",
                        type=str,
                        required=True,
                        help="Path to directory with training images")

    parser.add_argument("--reg_data_root",
                        type=str,
                        required=True,
                        help="Path to directory with regularization images")

    parser.add_argument("--embedding_manager_ckpt",
                        type=str,
                        default="",
                        help="Initialize embedding manager from a checkpoint")

    parser.add_argument("--class_word",
                        type=str,
                        default="dog",
                        help="Placeholder token which will be used to denote the concept in future prompts")

    parser.add_argument("--init_word",
                        type=str,
                        help="Word to use as source for initial token embedding")

    return parser


def train(model, optimizer, dataloaders, epochs):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0
    # training
    with tqdm(range(1, epochs + 1), ncols=100, ascii=True) as tq:
        for epoch in tq:
            model.train()

            total_count = torch.tensor([0.0])
            correct_count = torch.tensor([0.0])
            batch_time = time.time(); iter_time = time.time()
            for i, data in enumerate(trainloader):
                # Only works for batch_number == 1.
                imgs, _, label = data
                imgs, label = imgs.to(device), label.to(device)


                next_paf, next_heatmap, pre_paf, pre_heatmap = model(imgs)
                #                 print(next_heatmap[:, -1, :, :, :].view(-1,19,28*28).shape,label.view(-1, 28*28).shape)
                loss_ce = criterion_ce(next_heatmap[:, -1, :, :, :].view(-1,19,28*28), label.view(-1, 28*28).long())

                loss_mse_paf = criterion_mse(next_paf[:, :-1, :, :, :], pre_paf[:, 1:, :, :, :])
                loss_mse_hm = criterion_mse(next_heatmap[:, :-1, :, :, :], pre_heatmap[:, 1:, :, :, :])

                loss = loss_mse_paf + loss_mse_hm + 0.3 * loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_description(
                    'Train Epoch: {} [{}/{} ]\t Loss: {:.6f}'.format(epoch, i * len(imgs),
                                                                     len(trainloader.dataset), loss.item()))
                if i % 100 == 0 and i != 0:
                    print('')
                    print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                                                                               time.time()-iter_time, loss.item()))
                    iter_time = time.time()


            model_parameter_checkpoint_path = f'./ckpt/{epoch}_parameter_checkpoint.pth'
            opt_parameter_checkpoint_path = f'./ckpt/{epoch}_opt_parameter_checkpoint.pth'
            # 1) Saving all learnable parameters of the model and optimizer
            torch.save(model.state_dict(), model_parameter_checkpoint_path)
            torch.save(optimizer.state_dict(), opt_parameter_checkpoint_path)






def freeze_weights(model):
    for idx,key in enumerate(model['state_dict'].keys()):
        if "diffusion_model.out" not in key:
            model['state_dict'][key].requires_grad=False
        else:
            model['state_dict'][key].requires_grad=True
            # print(model['state_dict'][key].requires_grad)
        # print(idx,key,model['state_dict'][key].requires_grad,"diffusion_model.out" in key)

    return model





if __name__ == '__main__':
    # PATH VARS ENVS #
    # PATH = "/home/featurize/data/data/image"
    model_path ='D:\COMP-PROJECT-PROKECT\REPO\Dreambooth-Stable-Diffusion\models\ldm\stable-diffusion-v1\model.ckpt'
    batch_size = 1
    epochs = 10
    base_lr = 5e-5
    lr_cos = lambda n: 0.5 * (1 + np.cos(n / epochs * np.pi)) * base_lr


    model = torch.load(model_path)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    print(model.keys())
    print(model['state_dict'].keys())
    # init and save configs
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)
    # parser = argparse.

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        print(opt.base)
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""

        if opt.datadir_in_name:
            now = os.path.basename(os.path.normpath(opt.data_root)) + now

        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    # for k in nondefault_trainer_args(opt):
    #     trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    # trainer_config["accelerator"]='dp'
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    #DATA
    config.data.params.train.params.data_root = opt.data_root
    config.data.params.reg.params.data_root = opt.reg_data_root
    config.data.params.validation.params.data_root = opt.data_root
    data = instantiate_from_config(config.data)

    # data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # print(data.datasets.keys())
    #
    # # frozen weights
    # print(type(model))
    model = freeze_weights(model)

    #optimizer default
    # optim_state_dict = dict()
    # for key,weight in model['state_dict'].items():
    #     if weight.requires_grad:
    #         optim_state_dict[key] = weight
    # print([ if weight.requires_grad else for key,weight in model['state_dict'].items()])
    # print(model['state_dict'])
    # print(optim_state_dict)
    # print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9,0.999))



    # train(model, optimizer, dataloaders, epochs=epochs)
    print('training finished.')








