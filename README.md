# SpaGenFuse: A deep generative model for unified latent representation of spatial multi-omics data

## Overview
We present SpaGenFuse, a deep generative model for spatial multi-omics integration. By combining modality-specific variational autoencoders with Product-of-Experts fusion and graph structure learning, SpaGenFuse addresses inter-modality discrepancies to align latent representations and capture spatial architecture. The model avoids strong spatial smoothness assumptions, allowing for the precise identification of tissue structures. Extensive benchmarking confirms that SpaGenFuse outperforms existing methods in spatial domain identification, accurately revealing complex molecular patterns and tissue organization across diverse datasets and heterogeneous environments.
![SpaGenFuse model structure](./main.png)
> **Fig. 1: Overview of SpaGenFuse.** (A) Multi-omics Input. The spatial multi-omics data input module, which can take an arbitrary number of omics profiles as input. (B) Training. SpaGenFuse training consists of two stages, namely Stage I: Pre-training and Stage II: Integration. In Stage I, to learn the importance of each omics modality, the model trains an omics-specific variational autoencoder for each modality, and selects a corresponding distribution according to the statistical characteristics of the omics data to construct the decoder network. Stage II builds upon the omics-specific variational autoencoders and introduces a PoE-guided integration module in the latent space: (a) FusionUnit. Before PoE integration, to coordinate differences between different modalities, the omics-specific latent distributions are first mapped into a shared latent space through the Align Encoder, and then (b) Spectral Clustering and (c) Consistent Regularization are applied to learn the inherent graph structure within each omics and the consistent expression of different omics data in the shared latent space, respectively. Subsequently, PoE integration is used to obtain a unified representation, which is then mapped back to omics-specific latent distributions through the Align Decoder. Finally, a discriminator is introduced after the VAE decoder, and adversarial generative learning is used to improve the quality of data reconstruction. (C) Tasks. Downstream tasks. The unified representation obtained by PoE integration can be used for multiple downstream analyses, such as spatial domain identification / cell type identification, visualization, and differential expression analysis.

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
