import os

from blast_plot import blastn_evaluation

project_path = "/home/cxm/train/seq-exp/"
# nat_blast = os.path.join(project_path, 'data/blastdata/realBresult.txt')
STR_blast = os.path.join(project_path, 'data/blastdata/STRresult.txt')
Alper_blast = os.path.join(project_path, 'data/blastdata/Alperresult.txt')
IGEM_blast = os.path.join(project_path, 'data/blastdata/IGEMresult.txt')
ran_blast = os.path.join(project_path, 'data/blastdata/randomresult.txt')
# blastn_evaluation(nat_blast,gen_blast, ran_blast, report_path="./results/")
blastn_evaluation(STR_blast, Alper_blast, IGEM_blast, ran_blast, report_path="./results/")