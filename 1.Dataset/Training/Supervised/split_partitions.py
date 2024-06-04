import argparse
import os
import pandas as pd


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed that will define the `train` and `test` partitions",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Portion of data used for the test set wrt the training. Default to 0.2.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./Partition",
        help="Output path where to save the corpus partition. Default",
    )
    args = parser.parse_args()
    return args


def export_json(df, name):
    df.to_parquet(name, index=False)


def split_partitions(args):
    df = pd.read_json("1.Dataset/Training/Supervised/full_supervised_corpus.json", orient="index")
    df = df.sample(frac=1, random_state=args['seed'])
    tot_el = df.shape[0]
    print(f"There are {tot_el:,} elements in total")
    train_df, test_df = (
        df.iloc[: int(tot_el * (1 - args['test_size']))],
        df.iloc[int(tot_el * (1 - args['test_size'])) :],
    )
    train_df.to_csv("1.Dataset/Training/Supervised/train5.csv")
    test_df.to_csv("1.Dataset/Training/Supervised/test5.csv")
    print(
        f"Of these:\n\t- {train_df.shape[0]:,} will be used for training\n\t- {test_df.shape[0]:,} will be used for test"
    )
    path_train = args.output_path + "/" + str(args.seed) 
    path_test = args.output_path + "/" + str(args.seed)
    os.makedirs(path_train, exist_ok = True) 
    os.makedirs(path_test, exist_ok = True) 
    export_json(train_df, (path_train + "/" + "sample_train_corpus.parquet"))
    export_json(test_df, (path_test + "/" + "sample_test_corpus.parquet"))


if __name__ == "__main__":
    arg = {"seed": 5,
           "test_size": 0.2,
           "output_path": "./Partition"}
    # split_partitions(parse_arguments())
    split_partitions(arg)
