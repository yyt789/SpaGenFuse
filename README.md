# UniSpa: A deep generative model for unified latent representation of spatial multi-omics data

![UniSpa model structure](./main.png)

## Overview
We present UniSpa, a deep generative model for spatial multi-omics integration. By combining modality-specific variational autoencoders with Product-of-Experts fusion and graph structure learning, UniSpa addresses inter-modality discrepancies to align latent representations and capture spatial architecture. The model avoids strong spatial smoothness assumptions, allowing for the precise identification of tissue structures. Extensive benchmarking confirms that UniSpa outperforms existing methods in spatial domain identification, accurately revealing complex molecular patterns and tissue organization across diverse datasets and heterogeneous environments.

## Requirements
- Python==3.9 
- torch>=2.4.0
- numpy==1.24.4
- pandas==2.2.3
- scanpy==1.10.1
- scikit-learn==1.2.0
- scipy==1.9.1
- anndata==0.10.7
- matplotlib==3.9.4
- h5py==3.12.1
- tqdm==4.64.1
- squidpy==1.3.1
- rpy2==3.5.13
- R==4.4.2

## Data

SPOTS Mouse Spleen Data(Ben-Chetrit et al., 2023) - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353

Spatial-epigenome-transcriptome Mouse Brain Data(Zhang et al., 2023) - https://web.atlasxomics.com/visualization/Fan

Stereo-CITE-seq Mouse Thymus Data、Human Lymph Node Data (10x Visium CytAssist)(Long et al., 2024) - https://zenodo.org/records/10362607

MISAR-seq mouse embryonic brain RNA-ATAC data(Jiang et al., 2023) - https://zenodo.org/records/7480069

Slide-tags human melanoma RNA-ATAC data - https://singlecell.broadinstitute.org/single_cell/study/SCP2176

## Tutorial

Tutorials will be added subsequently.
