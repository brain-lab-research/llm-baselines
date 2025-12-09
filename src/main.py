import argparse
import copy
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import config
import distributed
import wandb
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.adafactor import Adafactor
from optim.adammini import Adam_mini
from optim.ademamix import AdEMAMix
from optim.ademamix2 import AdEMAMix2
from optim.adopt import ADOPT
from optim.base import train
from optim.clipped import (AdagradClip, AdaGradClipDelayedEta, AdamClip,
                           AdamClipDelayedEta)
from optim.lamb import Lamb
from optim.lion import Lion
from optim.mars import MARS
from optim.muon import Muon
from optim.normalized import NormalizedSGD
from optim.lipschitz_analyzer import LipschitzAnalyzer
from optim.lipschitz_scheduler import LipschitzScheduler
from optim.prodigy import Prodigy
from optim.schedule import get_scheduler
from optim.schedulefree import AdamWScheduleFree, SGDScheduleFree
from optim.sgdf import SGDF
from optim.shampoo import DistributedShampoo
from optim.sign import Signum
from optim.soap import SOAP
from optim.sophia import SophiaG


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args, parser, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
            entity=args.wandb_entity,
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(
        args.device
    )  # todo: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {"parameters": params_cnt, "optimized_parameters": optimized_params_cnt}
        )

    args.world_size = distributed_backend.get_world_size()

    if args.opt in ["adamw", "adam"]:
        device_type = "cuda" if "cuda" in args.device else "cpu"
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    elif args.opt == "soap":
        opt = SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )
    elif args.opt == "muon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        opt = Muon(
            muon_params=param_list,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_params=None,
            adamw_lr=args.lr / args.muon_lr_factor,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "ademamix":
        opt = AdEMAMix(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "ademamix2":
        opt = AdEMAMix2(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lion":
        opt = Lion(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sf-adamw":
        opt = AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "sf-sgd":
        opt = SGDScheduleFree(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "adam-mini":
        opt = Adam_mini(
            device=args.device,
            world_size=args.world_size,
            named_parameters=model.named_parameters(),  # check
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            model_sharding=args.model_sharding,
            dim=args.n_embd,
            n_heads=args.n_head,
            n_kv_heads=args.n_kv_head,
            verbose=args.adam_mini_verbose,
        )
    elif args.opt == "signsgd":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=0.0,  # always use zero momentum because its signSGD
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "signum":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "sgdf":
        opt = SGDF(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "prodigy":
        opt = Prodigy(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.weight_decay,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            fsdp_in_use=args.prodigy_fsdp_in_use,
        )
    elif args.opt == "sophiag":
        opt = SophiaG(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rho=args.sophia_rho,
        )
    elif args.opt == "shampoo":
        opt = DistributedShampoo(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            precondition_frequency=args.precondition_frequency,
            weight_decay=args.weight_decay,
            use_decoupled_weight_decay=True,
            # grafting_config=AdamGraftingConfig(
            #     beta2=args.beta2,  # oroginally, the default value is 0.999
            #     epsilon=1e-8,
            # ),
        )
    elif args.opt == "adopt":
        opt = ADOPT(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.opt in [
        "clip-adagrad",
        "clip-adagrad-delay-eta",
        "clip-adam",
        "clip-adam-delay-eta",
    ]:
        clipped_adagrad_cfg = {
            "lr": args.lr,
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "clipping": args.clipping_type,
            "max_grad_norm": 1.0,
        }
        if args.opt == "clip-adagrad":
            opt = AdagradClip(**clipped_adagrad_cfg)
        clipped_adagrad_delay_eta_cfg = {
            **clipped_adagrad_cfg,
            "exp_avg_sq_value": 0.0001,
            "etta": args.clipping_eta,
        }
        if args.opt == "clip-adagrad-delay-eta":
            opt = AdaGradClipDelayedEta(**clipped_adagrad_delay_eta_cfg)
        clipped_adam_cfg = {
            **clipped_adagrad_cfg,
            "betas": (args.beta1, args.beta2),
            "correct_bias": args.correct_bias,
        }
        if args.opt == "clip-adam":
            opt = AdamClip(**clipped_adam_cfg)
        clipped_adam_delay_eta_cfg = {
            **clipped_adam_cfg,
            "exp_avg_sq_value": 0.00001,
            "etta": args.clipping_eta,
        }
        if args.opt == "clip-adam-delay-eta":
            opt = AdamClipDelayedEta(**clipped_adam_delay_eta_cfg)
    elif args.opt == "mars":
        opt = MARS(
            group_specs,
            lr=args.mars_lr,
            betas=(args.mars_beta1, args.mars_beta2),
            weight_decay=args.weight_decay,
            amsgrad=False,
            gamma=args.mars_vr_gamma,
            is_approx=args.mars_is_approx,
            mars_type=args.mars_type,
            optimize_1d=False,  # we set in order to optimize 1D parameters with AdamW
            lr_1d=args.lr,  # AdamW's lr when optimize_1d=False
            betas_1d=(args.beta1, args.beta2),  # AdamW's betas when optimize_1d=False
            weight_decay_1d=0.1,  # AdamW's weight decay
        )
    elif args.opt == "adafactor":
        opt = Adafactor(
            group_specs,
            lr=args.lr,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.beta1,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lamb":
        opt = Lamb(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            adam=False,
            bias_correction=args.lamb_use_bias_correction,
        )
    elif args.opt == "normalized-sgd":
        opt = NormalizedSGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=False,
        )
    else:
        opt = torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        if args.use_lip_warmup: args.warmup_steps = 0
        assert (
            args.warmup_steps < args.iterations
        ), "Warmup steps must be < iterations."

        scheduler = get_scheduler(opt, args, group_specs=group_specs)

        if args.use_lip_warmup:
            scheduler = LipschitzScheduler(
                optimizer=opt,
                K_0=args.lipschitz_K_0,
                K_1=args.lipschitz_K_1,
                K_rho=args.lipschitz_K_rho,
                rho=args.lipschitz_rho,
                loss_star=args.lipschitz_loss_star,
                min_lr=args.lr / args.div_factor,
                max_lr=args.lr,
                adjust_K=not args.lipschitz_not_adjust_K,
                lr=args.lr,
                max_steps=args.iterations,
                mode=args.lipschitz_mode,
                decay_scheduler=scheduler,
                decay_scheduler_args=args,
                decay_scheduler_group_specs=group_specs,
                target=args.lipschitz_target,
            )
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists() and not args.do_not_auto_resume:
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Create Lipschitz analyzer
    lipschitz_analyzer = LipschitzAnalyzer(
        enabled=args.analyze_lipschitz,
        weight_norm_type=args.weight_norm_type,
        fit_rho=args.fit_rho,
        rho=args.rho,
        f_star=args.f_star,
        results_dir=f"lip_points/{exp_name}" if args.analyze_lipschitz else None
    )

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
        lipschitz_analyzer=lipschitz_analyzer,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend=None,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "full_eval_at",
        "distributed_backend",
        "latest_ckpt_interval",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
        "wandb_run_prefix",
        "do_not_auto_resume",
        "log_interval",
        "analyze_lipschitz",
        "output_plot",
    ],
):
    # Get the default values
    defaults = vars(parser.parse_args([]))

    # rank = distributed_backend.rank

    # Generate the prefix with key arguments
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}_" + prefix
    # prefix = f"{args.batch_size}x{args.acc_steps}" + prefix

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)
