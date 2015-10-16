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
  -model $DEMODIR/output/training-file.mpi-$N.model.gz\
  -word_vector $DEMODIR/data/MixedDomain/MixedDomains.zh.vec\
  -lambda_reg 0.00006\
  -lambda_reo 0.88236\
  -lambda_rec 0.11764\
  -isTest 1\
  -m 100\
  -v 1
