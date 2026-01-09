# test.py(项目的根目录)

import argparse
from src.engine.test_engine import run_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--cfg",
            type = str,
            default = None,
            help = "YAML配置文件路径",
            )

    return  parser.parse_args()

def main():
    args = parse_args()
    run_test(args.cfg)


if __name__ == "__main__":
    main()
