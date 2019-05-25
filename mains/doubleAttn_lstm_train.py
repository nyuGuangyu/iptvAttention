import tensorflow as tf
import sys

dir = "/home/gl1257/iptv_attention/"
if dir not in sys.path:
    sys.path.append(dir)
from data_loader.datagen_lstm import DataGenerator
from models.doubleAttn_lstm import Attn_Fuse_Model
from trainers.doubleAttn_lstm import AttnTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from tensorflow.python import debug as tf_debug
import argparse


def main(_):
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)
        config_file = "/home/lgy/IPTV_new/iptv_attention/configs/double_attn_sw_title_train_config.json"
        config = process_config(config_file)
    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()

    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)

    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = Attn_Fuse_Model(data, config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = AttnTrainer(sess, model, config, logger, data)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training.")
    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
