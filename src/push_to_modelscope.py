import argparse
import os
from modelscope.hub.api import HubApi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--token", default=os.getenv("MODELSCOPE_TOKEN", ""))
    parser.add_argument("--visibility", type=int, default=5)
    parser.add_argument("--license", default="Apache License 2.0")
    parser.add_argument("--chinese-name", default=None)
    parser.add_argument("--commit-message", default="upload model")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--original-model-id", default=None)
    parser.add_argument("--ignore-file-pattern", nargs="*", default=None)
    parser.add_argument("--lfs-suffix", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.token:
        raise ValueError("MODELSCOPE_TOKEN 未设置，请使用 --token 或环境变量 MODELSCOPE_TOKEN")
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"模型目录不存在: {args.model_dir}")

    api = HubApi()
    api.login(args.token)
    api.push_model(
        model_id=args.model_id,
        model_dir=args.model_dir,
        visibility=args.visibility,
        license=args.license,
        chinese_name=args.chinese_name,
        commit_message=args.commit_message,
        tag=args.tag,
        revision=args.revision,
        original_model_id=args.original_model_id,
        ignore_file_pattern=args.ignore_file_pattern,
        lfs_suffix=args.lfs_suffix,
    )
    print(f"上传完成: {args.model_id}")


if __name__ == "__main__":
    main()
