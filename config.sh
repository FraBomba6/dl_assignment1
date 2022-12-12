#! /bin/sh
apt-get update -y
apt-get upgrade -y
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get install python3.10 -yy
dpkg-reconfigure --frontend noninteractive tzdata
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py
rm get-pip.py
pip install -r requirements.txt
pip install gdown
apt-get install unzip tmux htop -y
mkdir data
cd data
gdown 1LNKILhCM6hvC1unvxdt3mLa4jUEQvuIN
unzip -P dluniud2022 assignment_1.zip
cd ..
