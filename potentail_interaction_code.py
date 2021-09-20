# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def user_input():
    print("\nHow would you like to run this?","\n",
          "(Perturb initial conditions (pic), combine models (cm), different times(dt))\n")
    method = input("Enter desired method: ")
    
    valid_methods = ["pic", "cm", "dt"]
    
    while True:
        if method in valid_methods:
            print("\nokay")
            break
        else:
            print("\nPlease choose one of \"pic\", \"cm\", or \"dt\" ")
            method = input("Enter desired method: ")
            
    
    # metric to optimise
        
    return method


def confirm(ready, method_):
    print(ready, "I am ready to go, and my desired method is", method_)

ready_to_go = input("Ready to go? ")

while True:
    if ready_to_go == "yes":
        method = user_input()
        break
    elif ready_to_go == "no":
        print("okay then.")
        break
    else:
        ready_to_go = input("Ready to go? ")        
    
if ready_to_go == "yes":
    confirm(ready_to_go, method)
    
#yeprint(Ensemble(method, metric_to_optimi      ))


