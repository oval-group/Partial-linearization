This is implementation for the paper:

P. Mohapatra, P. K. Dokania, C. V. Jawahar and M. Pawan Kumar, Partial Linearization based Optimization for Multi-class SVM. European Conference on Computer Vision (ECCV), 2016


For running the demo use the following command (present in demo/pascal_action_classification/):

learnSSVM(outputFile,lambda,method,temperature,rseed,gamma_1Meps)

Example:

learnSSVM('path/to/outputfile',0.01,'bcpl',0.01,1,1)
