from trviz.main import TandemRepeatVizWorker
from trviz.utils import get_sample_and_sequence_from_fasta

fasta_file_path = "HPRC.IRF2BPL.repeat_1_500.fa"

sample_ids, tr_sequences = get_sample_and_sequence_from_fasta(fasta_file_path)

tr_visualizer = TandemRepeatVizWorker()

tr_visualizer.generate_trplot(
    tr_id="IRF2BPL_HPRC_1_500",
    sample_ids=sample_ids,
    tr_sequences=tr_sequences,
    motifs=["CAG", "GCC", "GCG", "GGC", "CAA"]
)
