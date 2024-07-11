from .HeroVAP import HeroVAP


def vap_standardized_partitions(partition_folder, vap_output_folder):
    vap = HeroVAP(
        partition_folder=partition_folder, vap_output_folder=vap_output_folder
    )
    vap.vap_partition_folder()
