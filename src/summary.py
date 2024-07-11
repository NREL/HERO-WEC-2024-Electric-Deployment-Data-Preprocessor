from .ModaqSummarize import ModaqSummarize


def summarize_vap_partition(partition_folder, summary_output_folder):
    summarize = ModaqSummarize(
        partition_folder=partition_folder, summary_output_folder=summary_output_folder
    )
    summarize.summarize_partition_folder()
