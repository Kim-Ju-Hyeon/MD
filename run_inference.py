import click
from runner.runner import Runner
import traceback
from utils.logger import setup_logging
import os
from glob import glob
from utils.train_helper import set_seed, mkdir, edict2dict
import datetime

import pytz
from easydict import EasyDict as edict
import yaml


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

    log_file = os.path.join(config.exp_sub_dir, "log_inference_{}.txt".format(config.seed))
    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.exp_name))

    try:
        my_runner = Runner(config=config)
        my_runner.test()

    except:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
