import click
from runner.runner import Runner
import traceback
from utils.logger import setup_logging
import os
from utils.train_helper import set_seed, mkdir, edict2dict
import datetime

import pytz
from easydict import EasyDict as edict
import yaml


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    hidden_channels_list = [16, 32, 64, 128]
    num_blocks_list = [1, 2, 4, 6]

    for hidden_channels in hidden_channels_list:
        for num_blocks in num_blocks_list:
            config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

            hyperparameter = f'hidden_channels_{hidden_channels}' \
                             f'__num_blocks_{num_blocks}'

            now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            sub_dir = '_'.join([hyperparameter, now.strftime('%m%d_%H%M%S')])
            config.seed = set_seed(config.seed)

            config.exp_name = config.exp_name

            config.exp_dir = os.path.join(config.exp_dir, str(config.exp_name))
            config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
            config.model_save = os.path.join(config.exp_sub_dir, "model_save")

            mkdir(config.model_save)


            config.model.hidden_channels = hidden_channels
            if config.model_name == 'DimeNet':
                config.model.num_blocks = num_blocks
            elif config.model_name == 'SphereNet':
                config.model.num_layers = num_blocks
            elif config.model_name == 'ComENet':
                config.model.num_layers = num_blocks

            save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
            yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

            log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
            logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
            logger.info("Writing log file to {}".format(log_file))
            logger.info("Exp instance id = {}".format(config.exp_name))

            try:
                my_runner = Runner(config=config)
                my_runner.train()
                my_runner.test()

            except:
                logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
