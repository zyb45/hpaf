#!/usr/bin/env python3
import argparse, json
from hpaf.core.io import load_config
from hpaf.llm.factory import make_vision_client
from hpaf.agents.program_agent import ProgramAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--atomic-task', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = make_vision_client(cfg.llm)
    agent = ProgramAgent(client, 'configs/prompts.yaml', 'configs/api_registry.yaml')
    out = agent.run(args.image, args.atomic_task)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
