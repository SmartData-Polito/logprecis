from tbparse import SummaryReader
log_dir = "2.Training/supervised_training/results/entity_classification/token/microsoft_codebert-base/codeBERT_f1_seed130_0.000005_loss/seed_130/logs"
reader = SummaryReader(log_dir)
df = reader.scalars
# print(df)
# df.to_markdown("text.md")
df.to_csv("Scalars.csv")