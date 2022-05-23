import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random

import wandb
import GPUtil

import gym
from gym.envs.registration import register

from tqdm import tqdm
tqdm.pandas()
import pyarrow.parquet as pq

# AgentDDPGText is a custom implementation which can be found at ./elegantrl/agent.py and ./elegantrl/textProcessor.py
from elegantrl.agent import AgentDDPGText
from elegantrl.run import *
from elegantrl.config import Arguments

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

register(
    id='textassettrading-v0',
    entry_point='environment:TextAssetTradingEnv',
)


gym.logger.set_level(40)

# setup config
ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']

train_start_date = pd.to_datetime("2017-06-01").date()
train_end_date = pd.to_datetime("2020-05-31").date()

results_df = pd.DataFrame()

price_df = pq.ParquetDataset("./data/Price.parquet", validate_schema=False, filters=[('cc', 'in', ccs)])
price_df = price_df.read(columns=["price"]).to_pandas()
price_df["date"] = pd.to_datetime(price_df["date"]).apply(lambda x: x.date())
price_df = price_df.pivot(index='date', columns='cc', values='price')

warm_up_window = 180
rolling_windows_size = 360
rolling_windows_shift = 90

if __name__ == "__main__":
    with wandb.init() as run:
        # check if gpu is available if not wait until it becomes available
        gpu_id = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.7, maxMemory=0.7)
        while len(gpu_id) < 1:
            print("waiting for gpu ...")
            time.sleep(180)
            gpu_id = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.7, maxMemory=0.7)
        gpu_id = gpu_id[0]

        param = wandb.config
        #param = {"source": "CoinTelegraph", "window_size": 3, "learning_rate": 0.0002651, "use_attention": False, "rnn_layers": 2, "linear_layers": 3, "use_price": True, "dropout": 1, "model": "bert-base-uncased", "epochs": 1}

        # ensure that the number of epochs is below 3
        if param["epochs"] > 3:
            raise Exception("Number of epochs needs to be three or less")

        # load Data
        text_df = pq.ParquetDataset("data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', param["source"])])

        #text_df = text_df.read(columns=["content_w2vSum"]).to_pandas()
        text_df = text_df.read(columns=["content_bertToken"]).to_pandas()

        text_df["date"] = pd.to_datetime(text_df["date"]).apply(lambda x: x.date())
        text_df = text_df.set_index("date").drop("source", axis=1)
        #text_df = pd.concat((text_df["content_w2vSum"].progress_apply(pd.Series), text_df.iloc[:, -1:]), axis=1)
        text_df.columns = [str(x) for x in text_df.columns]
        date_range = [x.date() for x in pd.date_range(train_start_date, train_end_date)]

        assert len(date_range) > warm_up_window + rolling_windows_size, "Given data has too few observations for defined rolling windows"

        windows = []
        window_start_idx = warm_up_window
        window_end_idx = warm_up_window + rolling_windows_size
        while window_end_idx <= len(date_range):
            windows.append((date_range[window_start_idx], date_range[window_end_idx]))
            window_start_idx += rolling_windows_shift
            window_end_idx = window_start_idx + rolling_windows_size

        metrics = {"cumReturn": [],
                   "sharpeRatio": [],
                   "sortinoRatio": [],
                   "calmarRatio": [],
                   "mdd": []}

        for i, (test_roll_start_date, test_roll_end_date) in enumerate(windows):

            # create training and evaluation environments
            env = gym.make('textassettrading-v0',
                            price_df=price_df,
                            window_size= param["window_size"],
                            text_df=text_df, #num_df=text_df
                            timeframe_bound=(train_start_date, test_roll_start_date),
                            reward=None, #reward="sharpeRatio"
                            max_texts_per_day=30,
                            max_words_per_text=20,
                            turbulence_threshold=np.inf,
                            trade_fee_bid_percent=0.005,
                            trade_fee_ask_percent=0.005,
                            force_trades=False,
                            provide_actions=False,
                            target_return= 2)

            test_env = gym.make('textassettrading-v0',
                            price_df=price_df,
                            window_size= param["window_size"],
                            text_df=text_df, #num_df=text_df
                            timeframe_bound=(test_roll_start_date, test_roll_end_date),
                            reward= env.reward_type,
                            max_texts_per_day= env.texts_per_day,
                            max_words_per_text= env.words_per_text,
                            turbulence_threshold= env.turbulence_threshold,
                            trade_fee_bid_percent= env.trade_fee_bid_percent,
                            trade_fee_ask_percent= env.trade_fee_ask_percent,
                            force_trades= env.force_trades,
                            provide_actions= env.provide_actions,
                            render_env= True)

            args = Arguments(AgentDDPGText, env=env, env_test= test_env)

            args.learner_gpus = gpu_id

            # setup parameters for rl agent
            args.target_step = args.max_step * 2
            args.gamma = 0.98
            args.eval_times = 3
            args.epochs = 2

            args.learning_rate = param["learning_rate"]

            # setup parameters for textprocessing part of rl model, same for actor and critic
            #args.__argsActor_text_processor = "w2vSumProcessor"
            args.__argsActor_text_processor = "bertRnnSeperateProcessor"
            args.__argsActor_use_price = param["use_price"]
            args.__argsActor_use_attention = param["use_attention"]
            args.__argsActor_rnn_layers = param["rnn_layers"]
            args.__argsActor_linear_layers = param["linear_layers"]
            args.__argsActor_dropout_prop = param["dropout"]
            args.__argsActor_bert_model = param["model"]

            #args.__argsActor_text_processor = "w2vSumProcessor"
            args.__argsCritic_text_processor = "bertRnnSeperateProcessor"
            args.__argsCritic_use_price = param["use_price"]
            args.__argsCritic_use_attention = param["use_attention"]
            args.__argsCritic_rnn_layers = param["rnn_layers"]
            args.__argsCritic_linear_layers = param["linear_layers"]
            args.__argsCritic_dropout_prop = param["dropout"]
            args.__argsCritic_bert_model = param["model"]

            # train and evaluate the model
            completed_test_env = train_evaluate_and_test(args, run)

            pd.DataFrame(completed_test_env.history).to_excel("./results/{}_CV".format(run.name) + "{}.xlsx".format(i), index=False)

            wandb_dict = completed_test_env._total_ratios
            wandb_dict["cumReturn"] = completed_test_env._total_return
            wandb_dict = {k + '_CV': v for k, v in wandb_dict.items()}
            wandb.log(wandb_dict)

            for key in metrics.keys():
                if key == "cumReturn":
                    metrics[key].append(completed_test_env._total_return)
                else:
                    metrics[key].append(completed_test_env._total_ratios[key])

            completed_test_env.render_all(path= "./plots/{}_CV{}.png".format(run.name, i))

            del completed_test_env

        save_dict = {}
        for key in metrics.keys():
            save_dict[key] = np.nanmean(metrics[key])
            save_dict[key + "_std"] = np.nanstd(metrics[key])
            save_dict[key + "_q1"] = np.nanquantile(metrics[key], 0.25)
            save_dict[key + "_q3"] = np.nanquantile(metrics[key], 0.75)
            save_dict[key + "_min"] = np.nanmin(metrics[key])
            save_dict[key + "_max"] = np.nanmax(metrics[key])
            save_dict[key + "_med"] = np.nanmedian(metrics[key])

        wandb.log(save_dict)