#!/usr/bin/env python3
import argparse, json
from hpaf.core.io import load_config
from hpaf.llm.factory import make_vision_client
from hpaf.agents.task_agent import TaskAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = make_vision_client(cfg.llm)
    agent = TaskAgent(client, 'configs/prompts.yaml')
    out = agent.run(args.image, args.task)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
