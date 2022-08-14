# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm Ahmed')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import os
import nltk.data
from nltk import tokenize

sentecnesList =  []

for filename in os.listdir(r"D:\UDE\6th Semester\MEMS\MEWS Data\MEWS_Essays\MEWS_Essays\Essays_all\Schweiz\T1\test"):
   with open(os.path.join(r"D:\UDE\6th Semester\MEMS\MEWS Data\MEWS_Essays\MEWS_Essays\Essays_all\Schweiz\T1\test", filename)) as f:
       text = f.read()
       text = text.replace("ï»¿","")
       print(text)
       print ("##################### - End of thefile- ####################")
       print("------------------- The essays as sentences ---------");
       sentecnesList.append(tokenize.sent_tokenize(text))

print(len(sentecnesList))

