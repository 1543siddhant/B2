# If needed, uncomment to install:
# !pip install scipy statsmodels pandas numpy

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Simulated RNA-Seq Data Generation
np.random.seed(42)
genes = ["Gene1", "Gene2", "Gene3", "Gene4"]

# For each gene generate 10 replicates for Sample1 and Sample2
data = pd.DataFrame({
    "Gene": genes,
    "Sample1_Counts": [np.random.poisson(lam=100, size=10) for _ in genes],
    "Sample2_Counts": [np.random.poisson(lam=80, size=10) for _ in genes]
})

# Calculate mean counts for each gene
data["Sample1_Mean"] = data["Sample1_Counts"].apply(np.mean)
data["Sample2_Mean"] = data["Sample2_Counts"].apply(np.mean)

# Differential Expression Analysis using Welch's t-test (unequal variance)
p_values = []
for _, row in data.iterrows():
    stat, p_value = ttest_ind(row["Sample1_Counts"], row["Sample2_Counts"], equal_var=False)
    p_values.append(p_value)

data["PValue"] = p_values

# Correct p-values for multiple testing (FDR - Benjamini/Hochberg)
data["AdjPValue"] = multipletests(data["PValue"], method='fdr_bh')[1]

# Add log2 fold change (Sample2 vs Sample1). Add pseudocount 1 to avoid div0/log0.
data["Log2FC"] = np.log2((data["Sample2_Mean"] + 1) / (data["Sample1_Mean"] + 1))

# Identify Differentially Expressed Genes (example thresholds: adj p < 0.05 and |log2FC| >= 1)
differential_genes = data[(data["AdjPValue"] < 0.05) & (data["Log2FC"].abs() >= 1)].copy()

# Simulated Gene Ontology terms for demonstration
annotations = {
    "Gene1": "GO:0001234,GO:5678901",
    "Gene2": "GO:2345678,GO:8901234",
    "Gene3": "GO:1234567",
    "Gene4": "GO:5678901,GO:2345678"
}

# Map annotations (safe copy used above to avoid SettingWithCopyWarning)
differential_genes["GO_Annotations"] = differential_genes["Gene"].map(annotations)

# Build report
report_lines = []
report_lines.append("RNA-Seq Differential Expression Analysis Report\n")
report_lines.append("="*60)
report_lines.append("Summary of Analysis:\n")
report_lines.append("Two conditions were compared using simulated RNA-seq counts per gene (10 replicates each).\n")
report_lines.append("="*60 + "\n")

report_lines.append("All genes with basic stats:\n")
report_lines.append(f"{'Gene':<8} {'Mean_S1':>8} {'Mean_S2':>8} {'Log2FC':>8} {'AdjP':>10}")
report_lines.append("-"*60)
for _, row in data.iterrows():
    report_lines.append(f"{row['Gene']:<8} {row['Sample1_Mean']:8.2f} {row['Sample2_Mean']:8.2f} {row['Log2FC']:8.2f} {row['AdjPValue']:10.4f}")

report_lines.append("\n" + "="*60 + "\n")
report_lines.append("Differentially Expressed Genes (AdjP < 0.05 and |Log2FC| >= 1):\n")
if not differential_genes.empty:
    report_lines.append(f"{'Gene':<8} {'Log2FC':>8} {'AdjP':>10} {'GO_Annotations':>30}")
    report_lines.append("-"*80)
    for _, row in differential_genes.iterrows():
        report_lines.append(f"{row['Gene']:<8} {row['Log2FC']:8.2f} {row['AdjPValue']:10.4f} {row['GO_Annotations']}")
else:
    report_lines.append("No genes passed the thresholds (AdjP < 0.05 and |Log2FC| >= 1).\n")
    report_lines.append("You can relax thresholds or use a more powerful DE method (DESeq2/edgeR/limma-voom) for small sample sizes.\n")

# Save the report to a file
with open("RNA_SEQ_ANALYSIS.txt", "w") as report_file:
    report_file.write("\n".join(report_lines))

print("Analysis report generated as 'RNA_SEQ_ANALYSIS.txt'.")
print("\nBrief table (first lines):")
print("\n".join(report_lines[:20]))
