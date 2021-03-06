#!/bin/bash
if [ $# -ne 1 ]
then
  echo "Usage: $0 coreNum"
  exit -1
fi

N=$1
DEMODIR=`pwd -P`
export PYTHONPATH=$DEMODIR/src
mpirun -n $1 python $PYTHONPATH/nn/lbfgstrainer.py\
  -instances_of_Education $DEMODIR/data/Education/train/dev_trad_me_reorder.xml\
  -instances_of_Laws $DEMODIR/data/Laws/train/dev_trad_me_reorder.xml\
  -instances_of_Thesis $DEMODIR/data/Thesis/train/dev_trad_me_reorder.xml\
  -instances_of_Spoken $DEMODIR/data/Spoken/train/dev_trad_me_reorder.xml\
  -instances_of_Science $DEMODIR/data/Science/train/dev_trad_me_reorder.xml\
  -instances_of_News $DEMODIR/data/MixedDomain/test/nist03/trad_me_reorder.xml\
  -instances_of_Unlabel $DEMODIR/data/MixedDomain/dev/unlabel.src\
  -model $DEMODIR/output/training-file.mpi-$N.model.gz\
  -word_vector $DEMODIR/data/MixedDomain/MixedDomains.zh.vec\
  -lambda_rec 0.15\
  -lambda_reg 0.15\
  -lambda_reo 0.15\
  -lambda_unlabel 0.15\
  -isTest 0\
  -m 200\
  -v 0\
