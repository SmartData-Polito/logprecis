import pandas as pd
import matplotlib.pyplot as plt

NAME_TAG_loss = "Loss_epoch/validation"
FINAL_BEST_loss = 20
NAME_TAG_f1 = "macro_f1/validation"
FINAL_BEST_f1 = 25
NAME_TAG_loss_train = "Loss_epoch/train"

NAME_OUTPUT_IMAGE = "plot.png"
TITLE_GRAPH = "Loss vs F1 as stopping rule\nLR=5e-5 - seed=130"


df = pd.read_csv("Scalars.csv")
df_loss = df[df["tag"] == NAME_TAG_loss]
df_f1 = df[df["tag"] == NAME_TAG_f1]
df_loss_train =df[df["tag"] == NAME_TAG_loss_train]

# plt.plot(df_loss["step"], df_loss["value"], label = "Loss in validation")
# plt.plot(df_f1["step"], df_f1["value"],'c', label = "F1 score in validation")
# plt.axvline(x = FINAL_BEST_loss, color = 'blue', linestyle = '--', label = "Epoch of best model - loss")
# plt.axvline(x = FINAL_BEST_f1,color = 'c', linestyle = '--', label = "Epoch of best model - F1 score")
# plt.grid()
# plt.title(TITLE_GRAPH)
# plt.legend(fontsize="8.75")
# plt.savefig(NAME_OUTPUT_IMAGE)

list_ratio = [df_loss_train["value"].iloc[i]/df_loss["value"].iloc[i] for i in range(len(df_loss["step"]))]
plt.plot(df_loss["step"], list_ratio)
plt.plot(df_loss["step"], df_loss_train["value"], alpha=0.4, linestyle = '--')
plt.plot(df_loss["step"], df_loss["value"], alpha=0.4, linestyle = '--')
plt.savefig("plt_ratio.png")
