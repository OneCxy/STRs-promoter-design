# ProSTR
ProSTR is a two-stage framework for designing high-activity promoters by incorporating STR priors.<br>
In the first stage, TFBS elements and STR patterns are combined to form a core scaffold, and candidates with reasonable structures are selected.<br>
In the second stage, the upstream and downstream regions are generated based on the fixed scaffold to obtain full-length promoter sequences, while maintaining structural consistency.<br>
Finally, an independently trained activity predictor is used to score and rank the generated sequences, and high-potential candidates are selected.<br>
This “structure-first, then generation” strategy makes the process more controllable and improves both efficiency and stability.<br>

# Installation<br>
## Option 1: Conda<br>
conda env create -f environment.yml<br>
conda activate prostr<br>
## Option 2: pip<br>
pip install -r requirements.txt



# Prepare data
Reference:Johns N I, Gomes A L, Yim S S, et al.Metagenomic mining of regulatory elements enables programmable species-selective gene expression.Nature methods,2018, 15 (5): 323-329.

* ecoli_generation.csv      The data used by the generator <br>
* ecoli_prediction.csv      The data used by the predictor <br>

# Design Promoter Sequence
We take design promoters in E.coli as an example,to illustrate how to train the ProSTR model and design the promoter sequences.
## 1.Training the generator
* run \Generator\step1_cGAN.py    <br>

>>Train the first-step conditional generator to learn the baseline sequence distribution and produce initial candidates.<br>

* run \Generator\str.py    <br>

>>Introduce and fill STR patterns in generated candidates to build an STR-enriched candidate set.<br>

* run \Generator\loss.py   <br>

>>Compute similarity loss between generated and natural sequences for quality-aware filtering.<br>

* run \Generator\step2_cGAN.py <br>

>>Train the second-step generator on the filtered set to further improve sequence quality and functional characteristics.<br>
## 2. Training the predictor (can be done in advance)

run \prediction\GRU.py <br>
>>This step trains the activity predictor.<br>
A pretrained predictor is needed before generation and reused as the scoring module in both stages. It scores candidates after STR insertion and flanking completion, selects high-activity sequences for stage-two refinement, and filters final outputs by predicted activity, yielding a consistent and stable selection process.<br>
## 3.Evaluate the performance of ProSTR model generated sequences

* DNAshape:run valid\dnashape.py  <br>

* GC:run valid\GCviolin.py   <br>

* Diversity:run valid\edit1.py  run valid\geditdistence.py   <br>

* K-mer:run valid\kmer.py    <br>

* BLAST search:run valid\blast.py   <br>
