import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--data_root_dir", type=str, default="path/to/data_root_dir", help="Data directory to WSI features (extracted via CLAM")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible experiment (default: 1)")
    parser.add_argument("--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv")
    parser.add_argument("--dataset", type=str, default="tcga_blca_100", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="Path to latest checkpoint (default: none)")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="surmoe", help="Type of model (Default: surmoe)")
    parser.add_argument("--model_size",  type=str,choices=["small", "large"],     default="small", help="Size of some models (Transformer)")
    parser.add_argument("--modal", type=str, choices=["omic", "path", "pathomic", "cluster", "coattn"],default="coattn",help="Specifies which modalities to use / collate function in dataloader.")
    parser.add_argument("--fusion", type=str, choices=["concat", "bilinear"],default="concat",help="Modality fuison strategy")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam","AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="RAdam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine", "linear"], default="None")
    parser.add_argument("--num_epoch", type=int, default=30, help="Maximum number of epochs to train (default: 30)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function")
    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")
    parser.add_argument("--gene_dir", type=str, default="path/to/gene_root_dir", help="Data directory to gene")
    parser.add_argument("--num_pathway", type=int, default=30, help="Maximum number of pathways")

    args = parser.parse_args()
    return args
