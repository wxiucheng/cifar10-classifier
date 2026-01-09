# train.py(项目根目录)

import argparse
from src.engine.train_engine import run_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--cfg",
            type = str,
            default = "configs/simplecnn10_cifar10.yaml",
            help = "YAML配置文件路径",
            )
    return parser.parse_args()

def main():
    args = parse_args()
    run_train(args.cfg)


if __name__ == "__main__":
    main()
