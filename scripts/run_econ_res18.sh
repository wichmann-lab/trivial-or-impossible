#!/bin/bash

# How many conditions are there?
NUM_CONDITIONS=9

# Loop through all conditions
for COND in `seq 1 $NUM_CONDITIONS`
do

  # Re-set parameters for base network script
  MODEL="ResNet18"  # Model architecture: ConvNet, AltConvNet, ResNet18, DenseNet121
  DSET="ImageNet"  # Dataset: ARN, ImageNet
  INIT="same"  # Model initialisation: "same" or "different"
  OPTIM="SGD"  # Optimizer: "Adam" or "SGD"
  DATA="same"  # Training data: "same" or "different"
  ORDER="same"  # Order of training data: "same" or "different"
  LR=0.1  # Starting learning rate
  BATCHES="same"  # Batch size: "same" or "different"
  EPOCHS=90  # Training epochs
  CUDA=0  # CUDA randomness: 0 is deterministic and 1 is random
  VERBOSE=1  # Whether to print training output
  NUM_MODELS=5 # Number of models for this condition

  if [ $COND -eq 1 ] # 1: BASE condition add 10 epochs and only run for one model
  then
    EPOCHS=100
    NUM=1
    COND_NAME="Res18_Base_condition"
    sbatch ./conv_econ_base.sh "$MODEL" "$DSET" "$INIT" "$OPTIM" "$DATA" "$ORDER" "$LR" "$BATCHES" "$EPOCHS" "$CUDA" "$NUM" "$VERBOSE" "$COND_NAME"&
  elif [ $COND -eq 9 ] # 9: ARCHITECTURE condition with different architecture, set model flag
  then
    MODEL="DenseNet121"
    NUM=1
    COND_NAME="Res18_Different_architecture"
    sbatch ./conv_econ_base.sh "$MODEL" "$DSET" "$INIT" "$OPTIM" "$DATA" "$ORDER" "$LR" "$BATCHES" "$EPOCHS" "$CUDA" "$NUM" "$VERBOSE" "$COND_NAME"&
  elif [ $COND -eq 10 ] # 10: COMBINED condition with different order, optim, init, LR, architecture
  then
    ORDER="different"
    OPTIM="SGDN"
    INIT="different"
    LR=0.15
    MODEL="DenseNet121"
    COND_NAME="Res18_Combined_condition"
    for NUM in `seq 1 $NUM_MODELS`
    do
        sbatch ./conv_econ_base.sh "$MODEL" "$DSET" "$INIT" "$OPTIM" "$DATA" "$ORDER" "$LR" "$BATCHES" "$EPOCHS" "$CUDA" "$NUM" "$VERBOSE" "$COND_NAME"&
    done
  else

    # 2: OPTIM condition with adam, change optimizer string and learning rate
    if [ $COND -eq 2 ]
    then
      OPTIM="SGDN"
      NUM_MODELS=1
      COND_NAME="Res18_Different_optimizer"
    fi

    # 3: INIT condition with different initialisation, set init string
    if [ $COND -eq 3 ]
    then
      INIT="different"
      COND_NAME="Res18_Different_initialisation"
    fi

    # 4: CUDA condition with cuda randomness, set cuda flag
    if [ $COND -eq 4 ]
    then
      CUDA=1
      COND_NAME="Res18_CUDA_nondeterministic"
    fi

    # 5: LR condition with different LR, set LR value
    if [ $COND -eq 5 ]
    then
      LR=0.15
      COND_NAME="Res18_Different_LR"
    fi

    # 6: BATCHES condition with different batch size, set batch size
    if [ $COND -eq 6 ]
    then
      BATCHES="different"
      NUM_MODELS=1
      COND_NAME="Res18_Different_batchsize"
    fi

    # 7: ORDER condition with different training data order, set order flag
    if [ $COND -eq 7 ]
    then
      ORDER="different"
      COND_NAME="Res18_Different_dataorder"
    fi

    # 8: DATA condition with different training data, set data flag
    if [ $COND -eq 8 ]
    then
      DATA="different"
      NUM_MODELS=2
      COND_NAME="Res18_Different_data"
    fi

    # For each condition, NUM_MODELS number of models with these settings are run
    for NUM in `seq 1 $NUM_MODELS`
    do
        sbatch ./conv_econ.sh "$MODEL" "$DSET" "$INIT" "$OPTIM" "$DATA" "$ORDER" "$LR" "$BATCHES" "$EPOCHS" "$CUDA" "$NUM" "$VERBOSE" "$COND_NAME"&
    done

  fi # Finish for base condition if
done