def get_model_name(args):
    modelname = f"seg|{args.dataset}," + '-'.join([str(subject) for subject in args.subjects_for_test]) + f"|{args.n_val_subjects}|" \
                f"{args.window_size},{args.window_stride}|bs{args.batch_size}_lr{args.lr}_l1{args.weight_decayL1}_l2{args.weight_decayL2}|" + \
                f"{ '-'.join([str(rate) for rate in args.pooling])}_{args.latent_features}|" + \
                (f"ov{args.overlap}|" if args.overlap > 0 else "") + f"{args.arch}|cf{args.complexity_factor}_ks{args.kernel_size}|"

    return modelname[:-1]

def get_command(args):
    command = f"--dataset {args.dataset} --lr {args.lr} " + \
                "--subjects_for_test " + ' '.join([str(subject) for subject in args.subjects_for_test]) + " " \
                f"--window_size {args.window_size} --window_stride {args.window_stride} --batch_size {args.batch_size} " + \
                f"--arch {args.arch} --latent_features {args.latent_features} --pooling " + ' '.join([str(subject) for subject in args.pooling]) + " " + \
                f"--overlap {args.overlap} --complexity_factor {args.complexity_factor} --kernel_size {args.kernel_size} "

    command += f"--num_workers {args.num_workers} --max_epochs {args.max_epochs} --normalize --reduce_imbalance "
    command += f"--training_dir {args.training_dir} --n_val_subjects {args.n_val_subjects} "
    command += f"--weight_decayL1 {args.weight_decayL1} "
    command += f"--weight_decayL2 {args.weight_decayL2} "

    return command
