# If you haven't installed Biopython yet, uncomment the next line:
# !pip install biopython
'''1. Assignment: DNA Sequence Analysis. Task: Analyze a given DNA sequence and perform basic sequence 
manipulation, including finding motifs, calculating GC content, and identifying coding regions. Deliverable: A 
report summarizing the analysis results and any insights gained from the sequence.'''

from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction  # optional: we compute GC manually below for clarity

# Providing DNA Sequence (same as you gave)
dna_sequence = "AGTCAGTAGACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"

# Create Seq object (useful for translation, reverse_complement, etc.)
seq = Seq(dna_sequence)

# Function to find motifs in the sequence (returns 0-based positions)
def find_motifs(sequence, motif):
    # Ensure we operate on strings and use uppercase for robustness
    s = str(sequence).upper()
    m = motif.upper()
    positions = [i for i in range(len(s) - len(m) + 1) if s[i:i + len(m)] == m]
    return positions

# Function to identify coding regions (simple ORF finder on the + strand)
def identify_coding_regions(sequence):
    s = str(sequence).upper()
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    coding_regions = []

    i = 0
    # scan every nucleotide for potential start (allows overlapping ORFs)
    while i < len(s) - 2:
        if s[i:i + 3] == start_codon:
            start_index = i
            j = i + 3
            # step through in frame (codon-by-codon) looking for first in-frame stop
            while j < len(s) - 2:
                codon = s[j:j + 3]
                if codon in stop_codons:
                    stop_index = j + 3  # inclusive end index (stop codon end)
                    coding_regions.append((start_index, stop_index))
                    break
                j += 3
            # advance by 1 to allow overlapping starts
            i = start_index + 1
        else:
            i += 1

    return coding_regions

# (Optional) Find motifs on the reverse-complement strand too
def find_motifs_both_strands(sequence, motif):
    forward = find_motifs(sequence, motif)
    rc = find_motifs(sequence.reverse_complement(), motif)
    # convert reverse-complement positions to coordinates on forward strand:
    # if rc position is p on RC (0-based), the corresponding start on original seq is:
    # orig_pos = len(seq) - (p + len(motif))
    rc_on_forward = [len(sequence) - (p + len(motif)) for p in rc]
    return {'forward': forward, 'reverse_on_forward_coords': rc_on_forward}

# Example motifs
motif1 = "AGCTAGCTA"
motif2 = "CTAGCTAGC"
motif1_positions = find_motifs(seq, motif1)
motif2_positions = find_motifs(seq, motif2)

# Calculating GC content (explicit, avoids version differences)
s_upper = str(seq).upper()
gc_content = (s_upper.count('G') + s_upper.count('C')) / len(s_upper)

# Identifying coding regions
coding_regions = identify_coding_regions(seq)

# Create a human-readable report
report = "DNA Sequence Analysis Report\n\n"
report += f"Provided DNA Sequence (length {len(seq)}):\n{seq}\n\n"
report += "Analysis 1: Finding Motifs\n"
report += f"Motif 1 ({motif1}) found at 0-based positions: {motif1_positions}\n"
report += f"Motif 2 ({motif2}) found at 0-based positions: {motif2_positions}\n\n"
report += "Analysis 2: Calculating GC Content\n"
report += f"GC Content: {gc_content:.2%}\n\n"
report += "Analysis 3: Identifying Coding Regions (simple ORF scan on + strand)\n"
if len(coding_regions) > 0:
    report += "Coding regions found (0-based start, stop index is the first base after the stop codon):\n"
    for start, stop in coding_regions:
        # show translated AA sequence for the ORF (exclude stop codon translation '*' included by Biopython)
        orf_seq = seq[start:stop]
        aas = orf_seq.translate(to_stop=False)
        report += f"Start: {start}  Stop: {stop}  Length(nt): {len(orf_seq)}  AA (translated): {str(aas)}\n"
else:
    report += "No coding regions (ORFs starting with ATG and ending with TAA/TAG/TGA in-frame) found on + strand.\n"

# Save report to file
with open("DNA_SEQUENCE_ANALYSIS.txt", "w") as report_file:
    report_file.write(report)

print("Analysis report generated as 'DNA_SEQUENCE_ANALYSIS.txt'.")
print("\nSummary:")
print(report)
