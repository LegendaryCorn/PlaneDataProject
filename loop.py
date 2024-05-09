import subprocess
with open("output1.txt", "w+") as output:
    subprocess.call(["python", "./main.py"], stdout=output)

with open("output2.txt", "w+") as output:
    subprocess.call(["python", "./main.py"], stdout=output)

with open("output3.txt", "w+") as output:
    subprocess.call(["python", "./main.py"], stdout=output)

