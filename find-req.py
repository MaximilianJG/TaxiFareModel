import json 

with open("requirements.txt", "r") as file:
    req = file.read()
with open("remote_requirements.txt", "r") as file:
    rem = file.read()
req = req.split()
rem = rem.split()
# I think something more elegant than this split is possible
mm = {e.split("==")[0]:e.split("==")[1] for e in rem if "==" in e}
overlapping_req = { k:v for k, v in mm.items() if k in req }

with open("overlapping_req.txt", 'w') as file: 
    for key, value in overlapping_req.items():
        file.write(key + "==" + value + "\n")


    