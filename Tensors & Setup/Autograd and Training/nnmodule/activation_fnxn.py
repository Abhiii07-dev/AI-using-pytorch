"""
decides which data should move further and which will be blocked(Decision maker)
applies after a layer
linaer - activation - linear - activation - output

# Most used activation fxn ( nn.ReLU)
Rectified linear Unit :-
-ve no. ----> 0
+ve n0. -----> same


#nn.sigmoid (yes/No):
comnverts in 0 and 1
do not used in the middle layers ... only in the final output

# n.softmax (takes many numbers and convet rts them into one)



if : 
    hidden layers = ReLU
    binary output = sigmoid
    multiclass output = softmax


. input -> linear -> ReLU -> linear -> softmax / sigmoid -> output
"""
