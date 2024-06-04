import os

path = "/home/ec2-user/backend_aws/backend_docker"
if os.path.exists(path):
    print("Path exists")
else:
    print("Path does not exist")

