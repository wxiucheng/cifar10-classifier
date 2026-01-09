# demo_gradio.py(项目根目录)

from src.utils.args import parse_args
from src.engine.demo_gradio_engine import run_gradio


def main():
    args = parse_args()
    run_gradio(args.cfg)


if __name__ == "__main__":
    main()
