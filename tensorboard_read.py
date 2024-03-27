from tbparse import SummaryReader
log_dir = "2.Training/supervised_training/results_old/entity_classification/token/ehsanaghaei_SecureBERT/secureBERT_LR_00005"
reader = SummaryReader(log_dir)
df = reader.text
# print(df)
df.to_markdown("text.md")
# df.to_csv("Scalars.csv")