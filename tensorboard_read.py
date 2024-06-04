from tbparse import SummaryReader
log_dir = "2.Training/supervised_training/results/entity_classification/token/ehsanaghaei_SecureBERT/secureBERT_LR_000005/seed_777"
reader = SummaryReader(log_dir)
df = reader.text
# print(df)
df.to_markdown("text777.md")
# df.to_csv("Scalars.csv")