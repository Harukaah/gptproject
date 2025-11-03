# main.py
import argparse, subprocess, sys

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "generate"])
    ap.add_argument("--passthrough", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        cmd = [sys.executable, "train.py"] + args.passthrough
    else:
        cmd = [sys.executable, "generate.py"] + args.passthrough
    print(">>>", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
