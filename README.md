Algoritme om stadium 4 borstkanker te detecteren. 

Scores van algoritme voor maximale recall op stadium 4 identificatie gebruik makende van Random Forest
Accuracy: 0.9926339285714286 
Recall: 0.7222222222222222 
Precision: 0.3170731707317073 
F1-score: 0.44067796610169496
tn: 4434 fp: 28 
fn: 5 tp: 13
AUC of current model: 0.9902385576970966
![image](https://github.com/Simeonhm/brain_tumor_algorithm-/assets/145502662/2f47a62f-72f0-44a2-aefe-3a8c7dfeaeda)

Scores voor algoritme van elk stadium:
UITSLAGEN VOOR STADIUM: M
geen waardes bij stadium: M

UITSLAGEN VOOR STADIUM: 0
Accuracy: 1.0 
Recall: 1.0 
Precision: 1.0 
F1-score: 1.0
tn: 4479 fp: 0 
fn: 0 tp: 1
AUC of current model: 1.0

UITSLAGEN VOOR STADIUM: 1A
Accuracy: 0.9169642857142857 
Recall: 0.9890710382513661 
Precision: 0.8718043719896258 
F1-score: 0.9267428121307602
tn: 1755 fp: 346 
fn: 26 tp: 2353
AUC of current model: 0.9733467259430696

UITSLAGEN VOOR STADIUM: 1B
Accuracy: 0.9705357142857143 
Recall: 0.8306451612903226 
Precision: 0.48130841121495327 
F1-score: 0.6094674556213018
tn: 4245 fp: 111 
fn: 21 tp: 103
AUC of current model: 0.930083459225688

UITSLAGEN VOOR STADIUM: 1C
geen waardes bij stadium: 1C

UITSLAGEN VOOR STADIUM: 2A
Accuracy: 0.7154017857142857 
Recall: 0.9610062893081761 
Precision: 0.3804780876494024 
F1-score: 0.5451302176239744
tn: 2441 fp: 1244 
fn: 31 tp: 764
AUC of current model: 0.8545932430471996

UITSLAGEN VOOR STADIUM: 2B
Accuracy: 0.7424107142857143 
Recall: 0.9670731707317073 
Precision: 0.41302083333333334 
F1-score: 0.5788321167883211
tn: 2533 fp: 1127 
fn: 27 tp: 793
AUC of current model: 0.8679748100759694

UITSLAGEN VOOR STADIUM: 2C
geen waardes bij stadium: 2C

UITSLAGEN VOOR STADIUM: 3A
Accuracy: 0.9444196428571429 
Recall: 0.8761904761904762 
Precision: 0.2804878048780488 
F1-score: 0.4249422632794458
tn: 4139 fp: 236 
fn: 13 tp: 92
AUC of current model: 0.951033469387755

UITSLAGEN VOOR STADIUM: 3B
Accuracy: 0.9417410714285714 
Recall: 0.7850467289719626 
Precision: 0.2608695652173913 
F1-score: 0.3916083916083916
tn: 4135 fp: 238 
fn: 23 tp: 84
AUC of current model: 0.9435416136829441

UITSLAGEN VOOR STADIUM: 3C
Accuracy: 0.9421875 
Recall: 0.7758620689655172 
Precision: 0.2786377708978328 
F1-score: 0.41002277904328016
tn: 4131 fp: 233
fn: 26 tp: 90
AUC of current model: 0.9362989506621574

Algemene accuratie voorspellen stadium
0.7133928571428572

Voorbeeld van output
Echte waarde, voorspelling, correct?
 [2., 2., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [6., 6., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [6., 5., 0.],
 [2., 2., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [5., 6., 0.],
 [6., 6., 1.],
 [5., 5., 1.],
 [5., 6., 0.],
 [2., 2., 1.],
 [6., 5., 0.],
 [2., 6., 0.],
 [2., 2., 1.],
 [5., 5., 1.],
 [2., 2., 1.],
 [6., 6., 1.],
 [2., 2., 1.],
 [2., 2., 1.],
 [6., 2., 0.],
 [6., 5., 0.],
 [5., 5., 1.],
 [5., 6., 0.],
 [6., 5., 0.],
 [ 2., 2., 1.],
 [ 5., 5., 1.],
 [ 6., 6., 1.],
 [ 9., 10., 0.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 5., 6., 0.],
 [ 5., 6., 0.],
 [ 2., 2., 1.],
 [11., 11., 1.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 5., 6., 0.],
 [ 2., 2., 1.],
 [ 5., 5., 1.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 5., 5., 1.],
 [ 2., 2., 1.],
 [ 3., 8., 0.],
 [ 5., 6., 0.],
 [ 2., 2., 1.],
 [ 5., 5., 1.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 9., 9., 1.],
 [ 2., 2., 1.],
 [ 2., 2., 1.],
 [ 6., 5., 0.]])
