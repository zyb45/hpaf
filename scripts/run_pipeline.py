#!/usr/bin/env python3
import argparse
from hpaf.core.app import HPAFSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--mode', choices=['review', 'auto', 'manual'], default='manual')
    args = parser.parse_args()

    connect_robot = args.mode != 'manual'
    system = HPAFSystem.build(args.config, connect_robot=connect_robot)
    system.orchestrator.run(task_text=args.task, mode=args.mode)


if __name__ == '__main__':
    main()
